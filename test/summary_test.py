
import pytest
from utils.managers import SummaryDataCollections,TrainSummaryMaker,TestSummaryMaker
import tensorflow as tf 

images = SummaryDataCollections(summary_type='image',name='slice')
images['test1'] = tf.cast(tf.random.uniform(shape=[1,128,128,3],maxval=1.0)*255,dtype=tf.uint8)
images['test2'] = tf.cast(tf.random.uniform(shape=[1,128,128,3],maxval=1.0)*255,dtype=tf.uint8)
images['test3'] = tf.cast(tf.random.uniform(shape=[1,128,128,3],maxval=1.0)*255,dtype=tf.uint8)
buf = []
metrics = SummaryDataCollections(summary_type='scalar',name='psnr1')
metrics['test1'] = 2.0
metrics['test2'] = 3.0
metrics['test3'] = 5.0
metrics['test4'] = 7.0
buf.append(metrics)
metrics = SummaryDataCollections(summary_type='scalar',name='psnr2')
metrics['test1'] = 2.0
metrics['test2'] = 3.0
metrics['test3'] = 5.0
metrics['test4'] = 7.0
buf.append(metrics)
metrics = SummaryDataCollections(summary_type='scalar',name='psnr3')
metrics['test1'] = 2.0
metrics['test2'] = 3.0
metrics['test3'] = 5.0
metrics['test4'] = 7.0
buf.append(metrics)
metrics = SummaryDataCollections(summary_type='scalar',name='psnr4')
metrics['test1'] = 2.0
metrics['test2'] = 3.0
metrics['test3'] = 5.0
metrics['test4'] = 7.0
buf.append(metrics)
metrics = SummaryDataCollections(summary_type='scalar',name='ssim')
metrics['test1'] = 2.0
metrics['test2'] = 3.0
metrics['test3'] = 5.0
metrics['test4'] = 7.0
buf.append(metrics)
inputs = [images]+buf

@pytest.mark.parametrize('inputs', [inputs])
def test_summary_save(inputs):
    train_summary_maker = TrainSummaryMaker("./tmp")
    test_summary_maker = TestSummaryMaker("./tmp")
    train_summary_maker(inputs,tf.Variable(11,dtype=tf.int64))
    test_summary_maker(inputs,tf.Variable(13,dtype=tf.int64))

