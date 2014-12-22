from speedometer.main import start

start('test7_out.avi', skip=200, pos_x=-150, quality=0.01, save='test.mp4',
      multiprocessed=False, epochs=None, training_accuracy=30,
      training_length=2000, save_net='2000_frames_None_epochs_30_accuracy.xml')
