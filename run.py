from speedometer.main import start

start('test7_out.avi', skip=200, pos_x=-150, quality=0.01, save='test.mp4',
      multiprocessed=False, epochs=None, training_accuracy=30,
      speed_multi=0.0002, training_length=2000)
