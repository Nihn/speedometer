from speedometer.main import start

start('test6_out.avi', skip=100, pos_x=-150, quality=0.01, save='test.mp4',
      multiprocessed=False, epochs=3)
