import cv2
import numpy as np
import os


def analyze_image_features(image_path: str):
    # 1. טעינת התמונה
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # טוענים את התמונה (צבעונית לצורך תצוגה, אבל העיבוד נעשה על אפור)
    img_color = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # 2. יצירת הגלאי (The Detector)
    # אנחנו משתמשים ב-ORB, שמגביל את עצמו ל-500 הנקודות הכי חזקות
    orb = cv2.ORB_create(nfeatures=500)

    # 3. מציאת העוגנים והפקת תעודות הזהות (Detect & Compute)
    # kp (Keypoints) = רשימה של מיקומי הנקודות (איפה העוגן?)
    # des (Descriptors) = מטריצה של מספרים שמתארת כל נקודה (מי העוגן?)
    keypoints, descriptors = orb.detectAndCompute(img_gray, None)

    print(f"✅ Found {len(keypoints)} anchor points (features).")

    # 4. הצצה לנתונים ("מה המחשב רואה?")
    if len(keypoints) > 0:
        # ניקח את הנקודה הראשונה שנמצאה
        first_kp = keypoints[0]
        first_des = descriptors[0]

        print("\n--- Example of Anchor #1 ---")
        print(f"📍 Location (x, y): ({first_kp.pt[0]:.2f}, {first_kp.pt[1]:.2f})")
        print(f"📐 Orientation: {first_kp.angle:.2f} degrees")
        print(f"💪 Strength (Response): {first_kp.response:.2f}")

        print("\n💳 The 'ID Card' (Descriptor) for this point:")
        print(f"It is a list of {len(first_des)} numbers:")
        print(first_des)
        print("----------------------------\n")

    # 5. ויזואליזציה (ציור הנקודות על התמונה)
    # הדגל DRAW_RICH_KEYPOINTS מצייר עיגול בגודל שמייצג את גודל האזור שנבדק, וקו שמראה את הזווית
    img_with_features = cv2.drawKeypoints(
        img_color,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # הצגת התמונה
    cv2.imshow("Features (Anchors)", img_with_features)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

analyze_image_features("input_media\\pic_of_anchors.jpg")