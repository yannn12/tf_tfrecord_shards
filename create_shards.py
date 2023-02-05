images_list = os.listdir(images_path)
index = 0
n_images_shard = 800
n_shards = int(len(images_list) / n_images_shard) + (1 if len(images_list) % 800 != 0 else 0)

# tqdm is an amazing package that if you don't know yet you must check it
for shard in tqdm(range(n_shards)):
    # The original tfrecords_path is "{}_{}_{}.records" so the first parameter is the name of the dataset, 
    # the second is "train" or "val" or "test" and the last one the pattern.
    tfrecords_shard_path = tfrecords_path.format(dataset_name, train_val_test, '%.5d-of-%.5d' % (shard, n_shards - 1))
    end = index + n_images_shard if len(images_list) > (index + n_images_shard) else -1
    images_shard_list = images_list[index: end]
    with tf.io.TFRecordWriter(tfrecords_shard_path) as writer:
        for filename in images_shard_list:
            image_string = open(images_path + '/' + filename, 'rb').read()
            if filename in annotations_dict:
                annotations_list = annotations_dict[filename]
            else:
                annotations_list = [[0, 0, 0, 0, 0]]
            tf_example = image_example(image_string, annotations_list, filename)
            writer.write(tf_example.SerializeToString())
    index = end
