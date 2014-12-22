from speedometer.main import start

start('test7_out.avi', skip=300, pos_x=-150, quality=0.01, save='test.mp4',
      multiprocessed=False, epochs=None, training_accuracy=20,
      training_length=2000, save_net='2000_frames_4_epochs_None_accuracy.xml')
