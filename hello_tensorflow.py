import tensorflow as tf

#计算两个矩阵的乘积
marrix1 = tf.constant([[3, 3]])
marrix2 = tf.constant([[2], [2]])

product = tf.matmul(marrix1, marrix2)

# 第一种，需要手动关闭session
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()

# 第二种，和打开流一样的打开session，不需要关闭
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
