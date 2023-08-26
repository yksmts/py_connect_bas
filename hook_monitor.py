import cv2
import time
import numpy as np
from datetime import datetime


class HookMonitor:

    '''
    202308
    TOTOアクアテクノ向け引っ掛け監視システム
    '''

    new = False
    qr_code = ""
    qr_img = np.zeros((1000, 1000),np.uint8)

    def get_qr_code(self, img):
        
        # QRコードディテクターの作成
        qr_code_detector = cv2.QRCodeDetector()

        # QRコードの検出
        decoded_data, points, _ = qr_code_detector.detectAndDecode(img)

        # QRコードのデータを表示
        if decoded_data:
            color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
            # ポイントを整数に変換
            points = points.round().astype(int)
        
            # QRコードの位置を矩形で囲む
            for i in range(len(points)):
                cv2.polylines(color_img, [points[i]], isClosed=True, color=(0, 255, 0), thickness=2)

            # QRコードの内容をテキストとして表示
            text_position = (points[0][0][0], points[0][0][1] - 10)
            cv2.putText(color_img, decoded_data, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            self.new = False
            self.qr_img = color_img

            # QRコードの更新確認
            if self.qr_code != decoded_data:
                self.new = True
                self.qr_code = decoded_data
         
        else:
            self.new = False
            self.qr_img = img


    def save(self):

        now=datetime.now()
        formatted_date = now.strftime("%Y%m%d_%H%M%S")
        
        cv2.imwrite(formatted_date + self.qr_code + '_img.png', self.qr_img)


def reduce_domain(image, Row1, Column1, Row2, Column2):
    '''
    入力画像から矩形に画像を切り出して返す。
    '''
    
    return image[Row1:Row2, Column1:Column2]


def get_max_area(ROI):
    '''
    1.入力画像を2値化。
    2.ラベリングして最大ブロブを取得。
    3.最大ブロブのみ表示した2値化画像を返す。
    '''
    
    # ROI [Row1:Row2, Column1:Column2]
    height, width = ROI.shape[:2]    
    resizeROI = cv2.resize(ROI, (int(height * 0.1), int(width * 0.1)))

    # 画像を反転させる
    inverted_image = cv2.bitwise_not(resizeROI)

    # 2値化
    _, binary_image = cv2.threshold(inverted_image, 150, 255, cv2.THRESH_BINARY)

    # ブロブ検出
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

    # 最大のブロブを見つける
    max_area = -1
    max_label = -1
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_label = label

    # 最大ブロブの輪郭を抽出
    mask = np.uint8(labels == max_label)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    
    # 最大ブロブが白単色で塗りつぶされたブロブ画像を作成
    height, width = resizeROI.shape[:2]
    blob_image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.drawContours(blob_image, [largest_contour], 0, (255, 255, 255), -1)    
    blob_image = cv2.cvtColor(blob_image, cv2.COLOR_BGR2GRAY)

    return blob_image


def get_inner(image, ROI):
    '''
    輪郭内の最大内接円を探索し描画し、結果の画像と文字列を返す。
    '''
    
    height, width = image.shape[:2]

    # 輪郭を取得する
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
    # 輪郭までの距離を計算する
    dist_img = np.empty(image.shape, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            # この関数は，点が輪郭の内側にあるか，外側にあるか，
            # 輪郭上に乗っている（あるいは，頂点と一致している）かを判別。
            #
            # 返却値
            # 正値（内側），
            # 負置（外側）
            # 0（辺上）
            dist_img[i,j] = cv2.pointPolygonTest(contours[0], (j,i), True)

    # 最小値, 最大値, 最小値の座標, 最大値の座標
    minVal, maxVal, min_loc, max_loc = cv2.minMaxLoc(dist_img)
    minVal = abs(minVal)    # 外側
    maxVal = abs(maxVal)    # 内側

    # 距離イメージで求めた最大値,最小値の座標
    radius = int(maxVal)

    # 内接円画像
    inner_image = cv2.cvtColor(ROI, cv2.COLOR_GRAY2BGR)
    cv2.circle(inner_image, (max_loc[0] * 10,max_loc[1] * 10) , radius * 10, (0,0,255), 5, cv2.LINE_AA)
    
    return "内接円半径:{}".format(radius * 10), inner_image


def show_result(message, image):
    '''
    リサイズして結果表示
    '''

    # 画像をリサイズ
    resized = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)

    # 表示
    cv2.imshow("result", resized)
    print(message)
    

def online(exposure_time, flg):
    '''
    画像をカメラから取得して検査
    '''
    
    # カメラキャプチャのインスタンスを作成
    cap = cv2.VideoCapture(0)  # カメラ番号（0はデフォルトのカメラ）を指定

    # カメラが正常にオープンされたかを確認
    if not cap.isOpened():
        print("カメラがオープンできませんでした。")
        return
    
    # カメラの解像度を設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2048)

    # カメラのフレームレートを取得
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 自動露光
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

    # カメラの露光時間を設定
    # cap.set(cv2.CAP_PROP_EXPOSURE, exposure_time)

    # カメラからフレームを連続して読み込む
    while True:
        # カメラからフレームを読み込む
        ret, frame = cap.read()

        # 検査
        if flg == 1:
            # QRコードを画像読み取り
            message, image = get_qr_code(frame)
        else:
            # ROI取得
            ROI = reduce_domain(frame, 200, 500, 1900, 2200)

            # slect_shape_std
            gray_image = get_max_area(ROI)

            # inner
            message, image = get_inner(gray_image, ROI)

        # フレームの読み込みが成功したかを確認
        if not ret:
            print("フレームを読み込めませんでした。")
            break

        # 結果表示
        show_result(message, image)

        # 1秒待つ
        # time.sleep(0.05)

        # 'q'キーが押された場合、ループを終了
        if cv2.waitKey(1) == ord('q'):
            break

    # カメラキャプチャを解放
    cap.release()

    # ウィンドウを閉じる
    cv2.destroyAllWindows()

    print("画像の取り込みを終了しました。")


def offline(path, flg):
    '''
    オフライン処理
    '''
    
    # 画像の読み込み
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if flg == 1:
        # QRコードを画像読み取り
        message, image = get_qr_code(image)
    else:
        # ROI取得
        ROI = reduce_domain(image, 200, 500, 1900, 2200)

        # slect_shape_std
        gray_image = get_max_area(ROI)

        # inner
        message, image = get_inner(gray_image, ROI)
        
    # 結果表示
    show_result(message, image)


if __name__ == "__main__":
    
    # 画像の取り込みを実行
    # offline("NG.png", 2)

    online(10, 1)


