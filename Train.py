# *^_^* coding:utf-8 *^_^*
'''
ELEC/COMP 576 Final Project Programming
Yunda Jia, Siyao Xiao, Yu Wu, Rong Sun
'''
from __future__ import print_function
import cv2
import numpy as np
import tensorflow as tf
import random
from skimage import io
from skimage import filters
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


DEBUG = False
AVERAGE_S_THRESHOLD = 70
HSV_V_BLOCK_COUNT = 50
CANDIDATE_BLOCK_SIZE = 10
VIDEO_FILE = ["smoke1.avi", "smoke2.avi", "smoke3.avi", "smoke4.avi", "smoke5.avi", "smoke6.avi",
               "smoke7.avi", "smoke8.avi", "smoke9.avi", "smoke10.avi"]
BLOCK_WIDTH = 32
BLOCK_HEIGHT = 24
FRAME_SKIP = 1
FRAME_SIZE = (0, 0)
train_smoke_path = "medias/pictures/smoke_train_32x24/"
train_none_path = "medias/pictures/nosmoke_train_32x24/"
test_smoke_path = "medias/pictures/smoke_test_32x24/"
test_none_path = "medias/pictures/nosmoke_test_32x24/"


def get_move_toward(list_frames, m, n):
    """
    Get the direction of the area moving towards.
    list_frames is the list of current and last gray frame
    """
    bias = 2
    if m < bias or n < bias or m > FRAME_SIZE[0] - BLOCK_WIDTH - bias or n > FRAME_SIZE[1] - BLOCK_HEIGHT - bias:
        return 7
    block = list_frames[1][n:(n + BLOCK_HEIGHT), m:(m + BLOCK_WIDTH)]
    block1 = list_frames[0][(n - bias):(n + BLOCK_HEIGHT - bias), m:(m + BLOCK_WIDTH)]
    block2 = list_frames[0][(n - bias):(n + BLOCK_HEIGHT - bias), (m + bias):(m + BLOCK_WIDTH + bias)]
    block3 = list_frames[0][n:(n + BLOCK_HEIGHT), (m + bias):(m + BLOCK_WIDTH + bias)]
    block4 = list_frames[0][(n + bias):(n + BLOCK_HEIGHT + bias), (m + bias):(m + BLOCK_WIDTH + bias)]
    block5 = list_frames[0][(n + bias):(n + BLOCK_HEIGHT + bias), m:(m + BLOCK_WIDTH)]
    block6 = list_frames[0][(n + bias):(n + BLOCK_HEIGHT + bias), (m - bias):(m + BLOCK_WIDTH - bias)]
    block7 = list_frames[0][(n):(n + BLOCK_HEIGHT), (m - bias):(m + BLOCK_WIDTH - bias)]
    block8 = list_frames[0][(n - bias):(n + BLOCK_HEIGHT - bias), (m - bias):(m + BLOCK_WIDTH - bias)]

    list_result = []
    r1 = calc_direction(block, block1)
    list_result.append(r1)
    r2 = calc_direction(block, block2)
    list_result.append(r2)
    r3 = calc_direction(block, block3)
    list_result.append(r3)
    r4 = calc_direction(block, block4)
    list_result.append(r4)
    r5 = calc_direction(block, block5)
    list_result.append(r5)
    r6 = calc_direction(block, block6)
    list_result.append(r6)
    r7 = calc_direction(block, block7)
    list_result.append(r7)
    r8 = calc_direction(block, block8)
    list_result.append(r8)
    index = list_result.index(min(list_result))
    return index


def load_images(path):
    """
    load images from directory
    return a list of images data
    """
    img_list = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            img = io.imread(path + filename)
            img2 = filters.gaussian(img, sigma=1)
            img_flat = np.reshape(img2, (1, -1))
            img_list.append(img_flat)
    return img_list


def make_data():
    """
    make data of images
    return ndarray of image data
    """
    train_smoke_images = load_images(
        train_smoke_path
    )
    train_none_smoke_images = load_images(
        train_none_path
    )
    test_smoke_images = load_images(
        test_smoke_path
    )
    test_none_smoke_images = load_images(
        test_none_path
    )
    total_train_images_list = []
    total_train_labels_list = []  # labers
    total_test_images_list = []
    total_test_labels_list = []
    for i in range(len(train_smoke_images)):
        total_train_images_list.extend(
            np.array(
                train_smoke_images[i],
                dtype=np.float32
            )
        )
        total_train_labels_list.append([1, 0])
    for i in range(len(train_none_smoke_images)):
        total_train_images_list.extend(
            np.array(
                train_none_smoke_images[i],
                dtype=np.float32
            )
        )
        total_train_labels_list.append([0, 1])

    for i in range(len(test_smoke_images)):
        total_test_images_list.extend(test_smoke_images[i])
        total_test_labels_list.append([1, 0])
    for i in range(len(test_none_smoke_images)):
        total_test_images_list.extend(test_none_smoke_images[i])
        total_test_labels_list.append([0, 1])

    # transfrom image data from list to ndarray
    _total_train_images = np.array(total_train_images_list, dtype=np.float32)
    _total_train_labels = np.array(total_train_labels_list, dtype=np.float32)
    _total_test_images = np.array(total_test_images_list, dtype=np.float32)
    _total_test_labels = np.array(total_test_labels_list, dtype=np.float32)
    return _total_train_images, \
           _total_train_labels, \
           _total_test_images, \
           _total_test_labels


def calc_direction(list1, list2):
    if (len(list1) < 23 or (len(list2) < 23)):
        return 7
    if (len(list1[0]) < 23 or (len(list2[0]) < 23)):
        return 7
    s = 0.0
    for w in range(BLOCK_WIDTH):
        for h in range(BLOCK_HEIGHT):
            s += list1[h][w] - list2[h][w]
    s = s / (w * h)
    return s


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


if __name__ == "__main__":
    sess = tf.InteractiveSession()
    result_dir = './results/'  # directory where the results from the training are saved
    # tensorflow Variables
    xs = tf.placeholder(tf.float32, [None, 24 * 32 * 3], name='x')  # 32x24
    ys = tf.placeholder(tf.float32, [None, 2], name='y')
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 24, 32, 3])

    # conv1 layer #
    W_conv1 = weight_variable([5, 5, 3, 32])  # patch 5x5, in size 1, out size 32
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 32x24x32
    h_pool1 = max_pool_2x2(h_conv1)  # output size 16x12x32

    # conv2 layer #
    W_conv2 = weight_variable([5, 5, 32, 96])  # patch 5x5, in size 32, out size 64
    b_conv2 = bias_variable([96])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 16x12x64
    h_pool2 = max_pool_2x2(h_conv2)  # output size 8x6x64

    W_conv3 = weight_variable([5, 5, 96, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)  # output size 7x7x128

    # fc1 layer #
    W_fc1 = weight_variable([3 * 4 * 128, 1024])
    b_fc1 = bias_variable([1024])
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool3_flat = tf.reshape(h_pool3, [-1, 3 * 4 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # fc2 layer #
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='prediction')

    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(ys * tf.log(prediction),
                       1))  # loss
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    tf.summary.scalar(cross_entropy.op.name, cross_entropy)
    summary_op = tf.summary.merge_all()
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(result_dir, sess.graph)
    # load data
    total_train_images, \
    total_train_labels, \
    total_test_images, \
    total_test_labels = make_data()

    sess.run(init)

    round_num = 100
    max_step = 3000
    for index in range(max_step):
        batch_xs = []
        batch_ys = []
        for i in range(round_num):
            rand_num = random.randint(
                0,
                total_train_images.shape[0] - 1
            )
            batch_xs.append(total_train_images[rand_num])
            batch_ys.append(total_train_labels[rand_num])
        summary_str = sess.run(
            summary_op,
            feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.9}
        )
        if index % round_num == 0:
            test_accuracy = compute_accuracy(
                total_test_images, total_test_labels)
            print("Testing step: {}, Testing accuracy: {:.2f} ".format(index, test_accuracy))
            summary_writer.add_summary(summary_str, index)
            summary_writer.flush()
        # save the checkpoints every 1100 iterations
        if index % 300 == 0 or index == max_step:
            checkpoint_file = os.path.join(result_dir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=index)
        train_step.run(feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.9})  # run one train_step
    for VIDEO_FILE in VIDEO_FILE:
        VIDEO_SAVE_PATH = "re" + VIDEO_FILE
        fp = open(VIDEO_SAVE_PATH + ".txt", "w")
        cap = cv2.VideoCapture(
            "medias/videos/" + VIDEO_FILE)
        ret, start_frame = cap.read()
        start_gray_frame = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
        fgbg = cv2.createBackgroundSubtractorMOG2(
            history=500,
            detectShadows=False
        )
        height, width = start_frame.shape[:2]
        FRAME_SIZE = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(VIDEO_SAVE_PATH, fourcc, 25.0, FRAME_SIZE)
        frame_count = 0

        # save all blocks of the frame in HSV color space
        HSV_V_all_block = []
        two_gray_frames = []
        while 1:
            ret, frame = cap.read()
            if frame is None:
                print("The End!")
                break

            smooth_kernel = np.ones((5, 5), np.float32) / 25
            smooth_frame = cv2.filter2D(frame, -1, smooth_kernel)

            gray_frame = cv2.cvtColor(smooth_frame, cv2.COLOR_BGR2GRAY)
            if len(two_gray_frames) > FRAME_SKIP:
                two_gray_frames.pop(0)
            two_gray_frames.append(gray_frame)

            hsv_frame = cv2.cvtColor(smooth_frame, cv2.COLOR_BGR2HSV_FULL)
            if DEBUG:
                cv2.imshow("gray_frame", gray_frame)
                cv2.imshow("hsv_frame", hsv_frame)

            # GMM
            fgmask = fgbg.apply(gray_frame)
            kernel1 = np.ones((5, 5), np.uint8)
            kernel2 = np.ones((3, 3), np.uint8)
            fgmask = cv2.erode(fgmask, kernel2)
            fgmask = cv2.dilate(fgmask, kernel1)
            ret, fgmask_bin = cv2.threshold(fgmask, 0, 1, cv2.THRESH_BINARY)
            if DEBUG:
                ret, fgmask_bin_show = cv2.threshold(
                    fgmask,
                    0,
                    255,
                    cv2.THRESH_BINARY
                )
                cv2.imshow("fgmask_bin", fgmask_bin_show)

            HSV_V_each_block = []
            HSV_V_50_block = np.array(0)
            for m in range(0, width, BLOCK_WIDTH):
                for n in range(0, height, BLOCK_HEIGHT):
                    fgmask_clip = fgmask_bin[n:(BLOCK_HEIGHT + n), m:(BLOCK_WIDTH + m)]
                    candidate_clip = hsv_frame[n:(BLOCK_HEIGHT + n), m:(BLOCK_WIDTH + m)]

                    # store V of each frames
                    HSV_V_each_block.append(np.average(candidate_clip[:, :, 2]))

                    # find the move clips
                    if fgmask_clip.any():
                        if DEBUG:
                            cv2.rectangle(frame, (m, n), (m + BLOCK_WIDTH, n + BLOCK_HEIGHT), (255, 0, 0))

                        # average of S
                        candidate_clip_S = candidate_clip[:, :, 1]
                        average_S = np.average(candidate_clip_S)

                        # average of V
                        candidate_clip_V = candidate_clip[:, :, 2]
                        average_V = np.average(candidate_clip_V)

                        # if average of S lower than threshold it maybe smoke area
                        if (average_S < AVERAGE_S_THRESHOLD):
                            if DEBUG:
                                cv2.rectangle(frame, (m, n), (m + BLOCK_WIDTH, n + BLOCK_HEIGHT), (0, 255, 0))

                            # the value of V in the smoke area is higher
                            HSV_V_all_block_ndarray = np.array(HSV_V_all_block)
                            if (frame_count > HSV_V_BLOCK_COUNT - 1):
                                HSV_V_50_block = HSV_V_all_block_ndarray[:, m // 20]
                            elif (frame_count > 0):
                                HSV_V_50_block = HSV_V_all_block_ndarray[:frame_count, m // 20]

                            if (np.average(HSV_V_50_block) - average_V < 0):
                                cv2.rectangle(frame, (m, n), (m + BLOCK_WIDTH, n + BLOCK_HEIGHT), (0, 0, 255))
                                candidate_block = frame[n:(n + BLOCK_HEIGHT), m:(m + BLOCK_WIDTH)]

                                if frame_count > FRAME_SKIP:
                                    # if the object moving upward
                                    toward_num = get_move_toward(two_gray_frames, m, n)
                                    if 1 < toward_num < 5:
                                        cv2.rectangle(frame, (m, n), (m + BLOCK_WIDTH, n + BLOCK_HEIGHT), (255, 0, 0))
                                        candidate_block2_flat = np.reshape(candidate_block, (1, -1))
                                        result = sess.run(prediction,
                                                          feed_dict={xs: candidate_block2_flat, keep_prob: 1})
                                        if result[0][0] > result[0][1]:
                                            str = "find smoke at: frame{}({},{})".format(frame_count, n, m)
                                            print(str)
                                            fp.writelines(str + "\n")
                                            cv2.rectangle(frame, (m, n), (m + BLOCK_WIDTH, n + BLOCK_HEIGHT),
                                                          (255, 255, 255))

            cv2.putText(frame, "frame{}".format(frame_count), (20, 20), 1, 1.0, (12, 55, 50))
            out.write(frame)
            cv2.imshow("frame", frame)

            # store V of 50 frames before current frame
            if frame_count > HSV_V_BLOCK_COUNT - 1:
                HSV_V_all_block.pop(0)
                HSV_V_all_block.append(HSV_V_each_block)
            else:
                HSV_V_all_block.append(HSV_V_each_block)
            frame_count += 1

            if (cv2.waitKey(1) & 0xFF) == 27:
                print("ESC")
                break
    fp.close()
    cap.release()
    cv2.destroyAllWindows()