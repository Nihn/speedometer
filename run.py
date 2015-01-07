from speedometer.main import start

start('../test7_out.avi', skip=120, pos_x=-80, pos_y=-50, quality=0.2,
      multiprocessed=False, epochs=None, training_accuracy=15,
      speed_multi=5, training_length=100, load_net='test_3_30.xml')
