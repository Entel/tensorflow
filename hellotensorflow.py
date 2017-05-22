import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])

#print matrix1, matrix2

product = tf.matmul(matrix1, matrix2)
#print product

sess = tf.Session()
'''
sess = tf.Session()
result = sess.run(product)
print(result)

sess.close()


with tf.Session() as sess:
 result = sess.run([product])
 print result
 print sess.run(product)
'''

result = sess.run(tf.random_uniform([1], -1.0, 1.0))
result0 = sess.run(tf.zeros([1]))
print result0

