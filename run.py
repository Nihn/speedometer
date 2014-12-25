from speedometer.main import start

start('../test7_out.avi', skip=200, pos_x=-150, quality=0.01,
      multiprocessed=False, epochs=None, training_accuracy=10,
      speed_multi=0.0002, training_length=100)
