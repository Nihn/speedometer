from speedometer.main import start

start('test7_out.avi', skip=400, pos_x=-150, quality=0.01, save='test.mp4',
      multiprocessed=False, epochs=4, training_accuracy=40, load_net='text.xml')
