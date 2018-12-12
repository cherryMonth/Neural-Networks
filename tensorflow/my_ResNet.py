import collections
import tensorflow as tf
import keras
from keras.datasets.cifar import load_batch
from keras.datasets.cifar10 import load_data

"""

Typical use:

   from tensorflow.contrib.slim.nets import resnet_v2

ResNet-101 for image classification into 1000 classes:

   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnet_v2.resnet_arg_scope(is_training)):
      net, end_points = resnet_v2.resnet_v2_101(inputs, 1000)

ResNet-101 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnet_v2.resnet_arg_scope(is_training)):
      net, end_points = resnet_v2.resnet_v2_101(inputs,
                                                21,
                                                global_pool=False,
                                                output_stride=16)
"""

slim = tf.contrib.slim


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """
    使用collections.namedtuple设计ResNet基本的Block模块组的named tuple
        只包含数据结构,包含具体方法
    需要传入三个参数[scope,unit_fn,args]

     以Block('block1',bottleneck,[(256,64,1)]x2 + [(256,64,2)])为例
     scope = 'block1' 这个Block的名称就是block1

     unit_fn = bottleneck,  就是ResNet的残差学习单元

     args = [(256,64,1)]x2 + [(256,64,2)]
     args是一个列表,每个元素都对应一个bottleneck残差学习单元
     前面两个元素都是(256,64,1),后一个元素是(256,64,2)
     每个元素都是一个三元的tuple,代表(depth,depth_bottleneck,stride)
     例如(256,64,2)代表构建的bottleneck残差学习单元(每个残差学习单元里面有三个卷积层)中,
     第三层输出通道数depth为256,前两层输出通道数depth_bottleneck为64,且中间层的步长stride为2.

     这个残差学习单元的结构为[(1x1/s1,64),(3x3/s2,64),(1x1/s1,256)]

    整个block1中有三个bottleneck残差学习单元,结构为
    [(1x1/s1,64),(3x3/s2,64),(1x1/s1,256)]
    [(1x1/s1,64),(3x3/s2,64),(1x1/s1,256)]
    [(1x1/s1,64),(3x3/s2,64),(1x1/s1,256)]
    """

    """
    A named tuple describing a ResNet block.
    Its parts are:
      scope: The scope of the `Block`.
      unit_fn: The ResNet unit function which takes as input a `Tensor` and
        returns another `Tensor` with the output of the ResNet unit.
      args: A list of length equal to the number of units in the `Block`. The list
        contains one (depth, depth_bottleneck, stride) tuple for each unit in the
        block to serve as argument to unit_fn.
    """


def subsample(inputs, factor, scope=None):
    '''
        降采样方法,如果factor=1,则不做修改返回inputs,不为1,则使用slim.max_pool2d最大池化实现,
    :param inputs:
    :param factor: 采样因子
    :param scope:
    :return:
    '''
    """Subsamples the input along the spatial dimensions.

    Args:
      inputs: A `Tensor` of size [batch, height_in, width_in, channels].
      factor: The subsampling factor.
      scope: Optional variable_scope.

    Returns:
      output: A `Tensor` of size [batch, height_out, width_out, channels] with the
        input, either intact (if factor == 1) or subsampled (if factor > 1).
    """
    if factor == 1:
        return inputs
    else:
        # 进行池化的数据
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    """
        如果步长为1,直接使用slim.conv2d,使用conv2d的padding='SAME'
        如果步长大于1,需要显式的填充0(size已经扩大了),在使用conv2d取padding='VALID'
         （或者先直接SAME,再调用上面的subsample下采样）
    :param inputs:  [batch, height_in, width_in, channels].
    :param num_outputs:  An integer, the number of output filters.
    :param kernel_size: An int with the kernel_size of the filters.
    :param stride: An integer, the output stride.
    :param scope:
    :return:
    """

    """Strided 2-D convolution with 'SAME' padding.

    When stride > 1, then we do explicit zero-padding, followed by conv2d with
    'VALID' padding.

    Note that

       net = conv2d_same(inputs, num_outputs, 3, stride=stride)

    is equivalent to

       net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
       net = subsample(net, factor=stride)

    whereas

       net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')

    is different when the input's height or width is even, which is why we add the
    current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

    Args:
      inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
      num_outputs: An integer, the number of output filters.
      kernel_size: An int with the kernel_size of the filters.
      stride: An integer, the output stride.
      rate: An integer, rate for atrous convolution.
      scope: Scope.

    Returns:
      output: A 4-D tensor of size [batch, height_out, width_out, channels] with
        the convolution output.
    """
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,
                           padding='SAME', scope=scope)
    else:
        # kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        # 前三个参数依次为网络的输入，输出的通道，卷积核大小，stride是做卷积时的步长
        # 一般来说除非强制指定strude[0]和stride[3]之外，默认其为1，此处如果stride为数值即指定是长宽步长
        # 卷积时feature map的数量和卷积核数量一致
        # 三个通道上每个通道对应多个卷积核，先将其他的忽略，只看w1即第一个卷积核
        # 那么在w1的某位置（i,j）处的值，是由三个通道上都在w1（i,j）处的卷积结果相加然后再取激活函数值得到的。
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           padding='VALID', scope=scope)


@slim.add_arg_scope
def stack_blocks_dense(net, blocks,
                       outputs_collections=None):
    '''
    定义堆叠Blocks函数,
    :param net:  为输入  [batch, height, width, channels]
    :param blocks:  blocks为之前定义好的Blocks的class的列表,
    :param outputs_collections: 用来收集各个end_points和collections
    :return:
        使用两层循环,逐个Block,逐个Residual unit堆叠
        先使用variable_scope将残差单元命名改为block/unit_%d的形式
        在第二层,我们拿到每个Blocks中的Residual Unit的args,并展开
        再使用unit_fn残差学习单元生成函数顺序地创建并连接所有的残差学习单元
        最后,我们使用slim.utils.collect_named_outputs函数将输出net添加到collection

    '''
    """Stacks ResNet `Blocks` and controls output feature density.

    First, this function creates scopes for the ResNet in the form of
    'block_name/unit_1', 'block_name/unit_2', etc.


    Args:
      net: A `Tensor` of size [batch, height, width, channels].
      blocks: A list of length equal to the number of ResNet `Blocks`. Each
        element is a ResNet `Block` object describing the units in the `Block`.
      outputs_collections: Collection to add the ResNet block outputs.

    Returns:
      net: Output tensor

    """
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:  # 根据block的名字返回对于的变量
            for i, unit in enumerate(block.args):  # 对该block的残差单元进行遍历
                # 一个block可以含有多个残差单元
                # 在当前版本中一个block含有多个残差单元
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    # unit_fn 为一个残差单元，通过当前残差结构可以算出输出值
                    net = block.unit_fn(net,
                                        depth=unit_depth,  # 当前残差结构第三层卷积的通道数
                                        depth_bottleneck=unit_depth_bottleneck,  # 前两层卷积的通道数
                                        stride=unit_stride)  # 第二层的步长，其他的默认为1
            # 返回最后的输出作为下一个block的输入
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net


"""
BN 算法
"""


def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    '''
        这里创建ResNet通过的arg_scope,用来定义某些函数的参数默认值
        先设置好BN的各项参数,然后通过slim.arg_scope将slim.conv2d的几个默认参数设置好:
    :param is_training:
    :param weight_decay:  权重衰减率
    :param batch_norm_decay: BN衰减率默认为0.997
    :param batch_norm_epsilon:
    :param batch_norm_scale:
    :return:
    '''
    """Defines the default ResNet arg scope.

    TODO(gpapan): The batch-normalization related default values above are
      appropriate for use in conjunction with the reference ResNet models
      released at https://github.com/KaimingHe/deep-residual-networks. When
      training ResNets from scratch, they might need to be tuned.

    Args:
      is_training: Whether or not we are training the parameters in the batch
        normalization layers of the model.
      weight_decay: The weight decay to use for regularizing the model.
      batch_norm_decay: The moving average decay when estimating layer activation
        statistics in batch normalization.
      batch_norm_epsilon: Small constant to prevent division by zero when
        normalizing activations by their variance in batch normalization.
      batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
        activations in the batch normalization layer.

    Returns:
      An `arg_scope` to use for the resnet models.
    """
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    '''
        通过slim.arg_scope将slim.conv2d默认参数
        权重设置为L2正则
        权重初始化/激活函数设置/BN设置
    '''
    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            # The following implies padding='SAME' for pool1, which makes feature
            # alignment easier for dense prediction tasks. This is also used in
            # https://github.com/facebook/fb.resnet.torch. However the accompanying
            # code of 'Deep Residual Learning for Image Recognition' uses
            # padding='VALID' for pool1. You can switch to that choice by setting
            # slim.arg_scope([slim.max_pool2d], padding='VALID').
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride,
               outputs_collections=None, scope=None):
    '''
        bottleneck残差学习单元,这是ResNet V2论文中提到的Full Preactivation Residual Unit的
    一个变种, 它和V1中的残差学习单元的主要区别有两点:
        1. 在每一层前都用了Batch Normalization
        2. 对输入进行preactivation，而不是在卷积进行激活函数处理
    :param inputs:
    :param depth:
    :param depth_bottleneck:
    :param stride:
    :param outputs_collections:
    :param scope:
    :return:
    '''
    """Bottleneck residual unit variant with BN before convolutions.

    This is the full preactivation residual unit variant proposed in [2]. See
    Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
    variant which has an extra bottleneck layer.

    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.

    Args:
      inputs: A tensor of size [batch, height, width, channels].
      depth: The depth of the ResNet unit output.
      depth_bottleneck: The depth of the bottleneck layers.
      stride: The ResNet unit's stride. Determines the amount of downsampling of
        the units output compared to its input.
      rate: An integer, rate for atrous convolution.
      outputs_collections: Collection to add the ResNet unit output.
      scope: Optional variable_scope.

    Returns:
      The ResNet unit's output.
    """
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        # 获取输入的最后一个维度,即输出通道数
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        # 先做BN操作,在使用ReLU做preactivation
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        # 定义shortcut,如果残差单元的输入通道数depth_in和输出通道数depth一致,那么使用subsample
        # 按步长为stride对inputs进行空间上的降采样(确保空间尺寸和残差一致,因为残差中间那层的卷积步长为stride)
        # 如果输入/输出通道数不一样,我们用步长stride的1*1卷积改变其通道数,使得与输出通道数一致
        # 由于默认把三层结构中的前两层分别率定义为 1*1和3*3，所以只需要指定第三层的通道数即可
        if depth == depth_in:
            # 降采样，池化处理，根据残差单元中间层的步长来对输入进行池化，此时也会改变分别率
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            # 如果不一样，则使用卷积的方法改变维度
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')
        # 然后定义residual,这里residual有3层,先是一个1*1尺寸/步长为1/输出通道数为depth_bottleneck的卷积
        # 然后是一个3*3尺寸 -->最后还是一个1*1
        # 最终得到的residual,注意最后一层没有正则化也没有激活函数
        # 最后将residual和shortcut相加,得到最后的output,再添加到collection

        # 此时preact的shape为 shape=(32, 56, 56, 64)
        # 此时residual(conv1)的shape为 shape=(32, 56, 56, 64)
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')
        # 此时residual(conv2)的shape为 shape=(32, 56, 56, 64)，此时stride为2
        residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                               scope='conv2')

        # 此层为conv3，shape不定，通道数是根据每个残差结构改变
        """
        残差结构是定义在block中的一个元组，block列表中有几个元组就有几个残差结构，一个残差结构为三层网络
        """
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv3')

        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.name,
                                                output)


def resnet_v2(inputs,
              blocks,
              num_classes=None,
              global_pool=True,
              include_root_block=True,
              reuse=None,
              scope=None):
    '''

    :param inputs:
    :param blocks:
    :param num_classes:
    :param global_pool:
    :param include_root_block:
    :param reuse:
    :param scope:
    :return:
    '''
    """Generator for v2 (preactivation) ResNet models.

    This function generates a family of ResNet v2 models. See the resnet_v2_*()
    methods for specific model instantiations, obtained by selecting different
    block instantiations that produce ResNets of various depths.


    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      blocks: A list of length equal to the number of ResNet blocks. Each element
        is a resnet_utils.Block object describing the units in the block.
      num_classes: Number of predicted classes for classification tasks. If None
        we return the features before the logit layer.
      include_root_block: If True, include the initial convolution followed by
        max-pooling, if False excludes it. If excluded, `inputs` should be the
        results of an activation-less convolution.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.


    Returns:
      net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
        If global_pool is False, then height_out and width_out are reduced by a
        factor of output_stride compared to the respective height_in and width_in,
        else both height_out and width_out equal one. If num_classes is None, then
        net is the output of the last ResNet block, potentially after global
        average pooling. If num_classes is not None, net contains the pre-softmax
        activations.
      end_points: A dictionary from components of the network to the corresponding
        activation.

    Raises:
      ValueError: If the target output_stride is not valid.
    """
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck,
                             stack_blocks_dense],
                            outputs_collections=end_points_collection):
            net = inputs
            if include_root_block:
                # We do not include batch normalization or activation functions in conv1
                # because the first ResNet unit will perform these. Cf. Appendix of [2].
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=None, normalizer_fn=None):
                    # 此时net shape: 32 * 112 * 112 * 64
                    # 对输入张量的通道扩充到64，对分别率缩小一半,多少个卷积核就有多少个输出通道
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                # net 经过最大池化之后分别率减小一半， shape=(32, 56, 56, 64)
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            net = stack_blocks_dense(net, blocks)  # 把输入代入网络块进行运算, shape=(32,7,7,2048)
            # This is needed because the pre-activation variant does not have batch
            # normalization or activation functions in the residual unit output. See
            # Appendix of [2].
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
            if global_pool:
                # Global average pooling.
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
            if num_classes is not None:
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope='logits')
            # Convert end_points_collection into a dictionary of end_points.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if num_classes is not None:
                net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                net = slim.fully_connected(net, num_classes, activation_fn=None, scope='result')
                end_points['predictions'] = slim.softmax(net, scope='predictions')
            return net, end_points


def resnet_v2_50(inputs,
                 num_classes=None,
                 global_pool=True,
                 reuse=None,
                 scope='resnet_v2_50'):
    """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        # scope un_fun  args
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_101(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_101'):
    """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_152(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_152'):
    """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_200(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_200'):
    """ResNet-200 model of [2]. See resnet_v2() for arg and return description."""
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)


def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test


def scheduler(epoch):
    if epoch < 81:
        return 0.1
    if epoch < 122:
        return 0.01
    return 0.001


batch_size = 32
import numpy as np

(x_train, y_train), (x_test, y_test) = load_data()  # 50000, 32,32,3
y_train = np.array(keras.utils.to_categorical(y_train, 10).tolist())
y_test = np.array(keras.utils.to_categorical(y_test, 10).tolist())
x_train, x_test = color_preprocessing(x_train, x_test)
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])

with slim.arg_scope(resnet_arg_scope(is_training=False)):
    net, end_points = resnet_v2_152(x, 10)  # 分辨率会缩小32倍

y_pred = end_points['predictions']
cross_entropy = -tf.reduce_sum(y * tf.log(y_pred))
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
n_epochs = 5

for epoch_i in range(n_epochs):
    train_accuracy = 0
    for k in range(len(x_train) // batch_size):
        result = sess.run([optimizer, accuracy],
                          feed_dict={x: x_train[k * batch_size:(k + 1) * batch_size],
                                     y: y_train[k * batch_size:(k + 1) * batch_size]})
        print(result)
