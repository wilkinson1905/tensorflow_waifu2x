# -*- coding: utf-8 -*-

# waifu2x.py by @marcan42 based on https://github.com/nagadomi/waifu2x
# MIT license, see https://github.com/nagadomi/waifu2x/blob/master/LICENSE

# 必要なモジュールのインポート
import json, sys, numpy as np # jsonサポート、システム、数値ユーティリティ(と思われる)
from scipy import misc, signal # scipyより、その他処理、信号処理
from PIL import Image # 画像ファイル取り扱いユーティリティ
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Model export script: https://mrcn.st/t/export_model.lua (needs a working waifu2x install)

#infile, outfile = sys.argv[1:] # コマンドライン引数読み込み(入力ファイル、出力ファイル、モデルjsonファイルへのパス)


# モデルの1階層が持つ情報:
#   * nInputPlane : このモデルに入力するべき平面の数
#   * nOutputPlane : このモデルから出力されるべき平面の数
#   * weight : 入力平面に対して畳み込み演算を行う重み行列の集合
#     - nInputPlane個の重み行列をパックしたものがnOutputPlane個入っている。行列の要素は全てfloat値。
#     - 例えば、初回は1個の重み行列をパックしたものが32個、2階層目は32個の重み行列をパックしたものが32個(つまり行列の数は1024個)入っている。
#     - 1つの重み行列は全て3x3行列となっている。従って、1回畳み込みを行う度に平面の大きさは上下左右1要素ずつ小さくなる。
#     - 計算時のアンカー(畳み込みの中心とする行列の位置)はおそらく(2,2)であると思われる(そうでないと、出力平面が変な方向に移動してしまうため)。
#   * bias : 出力平面一つに対して(全要素に加算することにより)掛けるバイアス値
#     - float値がnOutputPlaneの数だけ用意されている。
#   * kW,kH : 未使用値(重み行列1つの大きさであると思われる。配布されているモデルの場合は全て3x3である)

# im = Image.open(infile).convert("YCbCr") # 入力ファイルの読み込み -> YCbCr色空間への変換

# width = 2*im.size[0]
# height =  2*im.size[1]

# im = misc.fromimage(im.resize((width, height), resample=Image.NEAREST)).astype("float32")
# 入力画像を2倍の幅・高さにNearestNeighbor法でリサイズした後、それをscipyで取り扱い可能な行列表現にし、
# その要素の型を32bit-floatにする



def upscale_and_denoise(images, power = 1):#images:[batch,width,hight],Y of YUV power:the output size is 2^power times
    output_size = (images.shape[0], images.shape[1] * 2 ** power, images.shape[2] * 2 ** power)
    #定数定義
    scalemodelpath = "scale2.0x_model.json"
    denoisemodelpath = "noise3_model.json"
    #モデルの読み込み
    model_list = []
    model_list.append(json.load(open(scalemodelpath)))# jsonファイルのロード(jsonファイルに記述された構造をそのままメモリ上に展開していると思われる)
    model_list.append(json.load(open(denoisemodelpath)))
    images = np.array(images)
    padding_size = len(model_list[0]) + len(model_list[1])
    convolution_size = 3
    # planes = np.pad(images,((0,0),(convolution_size, convolution_size),(convolution_size,convolution_size)), "edge") / 255.0
    planes = images / 255.0
    planes = planes.reshape(planes.shape[0], planes.shape[1], planes.shape[2], 1)
    # 画像データの周りに、画像の端をコピーする形で、モデルの大きさ(核の行列の大きさ)分だけパッドを入れ、0~1の間でクリップする
    # このplanesは輝度情報のみを取り出している(!)
    # オリジナルのwaifu2xも、reconstruct時に入力をYUV色空間に変換した後、Yのみを取り出して処理している
    
    # count = sum(step["nInputPlane"] * step["nOutputPlane"] for step in model)# 畳み込み演算の必要回数を計算
    # つまり、countの数だけ入力平面に対する重み行列の畳み込みが行われる。
    planes = np.transpose(planes, (0, 1, 2, 3))
    progress = 0
    x = tf.constant(planes, dtype=tf.float32)
    for i in range(power):
        x = tf.image.resize_nearest_neighbor(x,[planes.shape[1] * 2 ** (i + 1), planes.shape[2] * 2 ** (i + 1)])
        x = tf.pad(x, [[0,0],[padding_size, padding_size], [padding_size, padding_size],[0, 0]], "SYMMETRIC")
        for model in model_list:
            for step in model: # ループ:ステップ(1つのモデル階層) 始め
    
                # assert step["nInputPlane"] == planes.shape[1]
                # このステップのモデルに定義された入力平面の数と実際の入力平面の数は一致していなければならない
                # assert step["nOutputPlane"] == len(step["weight"]) == len(step["bias"])
                # モデルの出力平面はモデルの重み行列集合の数とそのバイアスの数と一致していなければならない
                # つまり、各ステップの重み行列集合の数とそのバイアスの数だけ、そのステップによって平面が出力される
                # o_planes = [] # 出力平面の格納場所を初期化
                W = tf.constant(np.transpose(np.array(step["weight"]), (2, 3, 1, 0)), shape=(3, 3, step["nInputPlane"], step["nOutputPlane"]),dtype=tf.float32)
                b = tf.constant(np.array(step["bias"]), shape=(1, 1, 1, step["nOutputPlane"]), dtype=tf.float32)
                x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID", data_format="NHWC")
                x = x + b
                x = tf.maximum(x, 0.1 * x)
    with tf.Session(config = tf.ConfigProto(log_device_placement = False, gpu_options = tf.GPUOptions(allow_growth = False))) as sess:
        planes = sess.run(x)
    
    # ループ:ステップ 終わり
    planes = np.clip(planes, 0, 1) * 255
    planes = planes.reshape(output_size)
    return planes.astype(np.uint8)
    # 得られた出力平面の全要素を0~1にクリップした後、

input_image_list = []
split_size = (320, 180)
batch_size = 1
for i in range(1):# 1080 8G 25 * 346 * 484 
    im = Image.open("mini_magi.png").convert("YCbCr") # 入力ファイルの読み込み -> YCbCr色空間への変換
    width = im.size[0]
    height =  im.size[1]
    im_y = misc.fromimage(im).astype("float32")[:,:,0]
    scale = 4
    im = misc.fromimage(im.resize((width * scale, height * scale), resample=Image.NEAREST)).astype("float32")
    im_y_output = np.zeros(( height * scale, width * scale))
    print(im.shape, im_y_output.shape)
    input_image_list = []
    location_list = []
    for w in range(0, width, split_size[0]):
        for h in range(0,height, split_size[1]):
            print(w,h)
            input_image_list.append(im_y[h:h + split_size[1], w:w + split_size[0]])
            location_list.append([w,h])
            if len(input_image_list) >= batch_size:
                images = upscale_and_denoise(np.array(input_image_list), 2)
                for image,location in zip(images,location_list):
                    w = location[0]
                    h = location[1]
                    im_y_output[h * scale : (h + split_size[1]) * scale, w * scale :(w + split_size[0]) * scale ] = image
                input_image_list = []
                location_list = []
    im[:,:,0] = im_y_output
    misc.toimage(im, mode="YCbCr").convert("RGB").save("out.png")
    sys.stderr.write("Done\n")
    sys.exit(0)
# images = upscale_and_denoise(np.array([[[1,1,1],[1,1,1],[1,1,1]],[[2,2,2],[2,2,2],[2,2,2]]]))
# images = upscale_and_denoise(np.array(input_image_list))
print(images.shape)

misc.toimage(im, mode="YCbCr").convert("RGB").save(outfile)
sys.stderr.write("Done\n")
