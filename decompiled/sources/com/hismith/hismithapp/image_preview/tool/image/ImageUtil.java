package com.hismith.hismithapp.image_preview.tool.image;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.os.Environment;
import android.text.TextUtils;
import com.hismith.hismithapp.image_preview.tool.common.Print;
import com.hismith.hismithapp.image_preview.tool.ui.PhoneUtil;
import com.luck.picture.lib.config.PictureMimeType;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

/* loaded from: classes3.dex */
public class ImageUtil {
    private static final String TAG = "ImageUtil";

    public static int getBitmapDegree(String str) {
        try {
            int attributeInt = new ExifInterface(str).getAttributeInt(androidx.exifinterface.media.ExifInterface.TAG_ORIENTATION, 1);
            if (attributeInt == 3) {
                return 180;
            }
            if (attributeInt != 6) {
                return attributeInt != 8 ? 0 : 270;
            }
            return 90;
        } catch (IOException e) {
            e.printStackTrace();
            return 0;
        }
    }

    /* JADX WARN: Can't wrap try/catch for region: R(11:0|1|(2:3|(8:5|6|(1:8)|9|(1:13)|14|15|(2:17|18)(1:20)))|(1:32)(1:31)|6|(0)|9|(2:11|13)|14|15|(0)(0)) */
    /* JADX WARN: Code restructure failed: missing block: B:22:0x0050, code lost:
    
        java.lang.Runtime.getRuntime().gc();
     */
    /* JADX WARN: Code restructure failed: missing block: B:23:0x0058, code lost:
    
        if (r2 != null) goto L25;
     */
    /* JADX WARN: Code restructure failed: missing block: B:26:0x0060, code lost:
    
        r2.recycle();
     */
    /* JADX WARN: Code restructure failed: missing block: B:27:0x0063, code lost:
    
        java.lang.Runtime.getRuntime().gc();
     */
    /* JADX WARN: Removed duplicated region for block: B:17:0x006c  */
    /* JADX WARN: Removed duplicated region for block: B:20:? A[RETURN, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:8:0x0032  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static Bitmap getImageBitmap(String str, float f, float f2) {
        float f3;
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        Bitmap decodeFile = BitmapFactory.decodeFile(str, options);
        options.inJustDecodeBounds = false;
        int i = options.outWidth;
        int i2 = options.outHeight;
        if (i > i2) {
            float f4 = i;
            if (f4 > f) {
                f3 = f4 / f;
                options.inSampleSize = (int) (f3 > 0.0f ? f3 : 1.0f);
                options.inPreferredConfig = Bitmap.Config.ARGB_8888;
                options.inDither = false;
                options.inPurgeable = true;
                options.inInputShareable = true;
                if (decodeFile != null && !decodeFile.isRecycled()) {
                    decodeFile.recycle();
                }
                decodeFile = BitmapFactory.decodeFile(str, options);
                return decodeFile == null ? rotateBitmapByDegree(decodeFile, getBitmapDegree(str)) : decodeFile;
            }
        }
        f3 = (i >= i2 || ((float) i2) <= f2) ? 1.0f : options.outHeight / f2;
        options.inSampleSize = (int) (f3 > 0.0f ? f3 : 1.0f);
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        options.inDither = false;
        options.inPurgeable = true;
        options.inInputShareable = true;
        if (decodeFile != null) {
            decodeFile.recycle();
        }
        decodeFile = BitmapFactory.decodeFile(str, options);
        if (decodeFile == null) {
        }
    }

    public static Bitmap getImageBitmap(String str, int i) {
        BitmapFactory.Options options = new BitmapFactory.Options();
        boolean z = true;
        options.inJustDecodeBounds = true;
        Bitmap decodeFile = BitmapFactory.decodeFile(str, options);
        options.inJustDecodeBounds = false;
        options.inSampleSize = 1;
        options.inPreferredConfig = Bitmap.Config.RGB_565;
        options.inDither = false;
        options.inPurgeable = true;
        options.inInputShareable = true;
        if (decodeFile != null && !decodeFile.isRecycled()) {
            decodeFile.recycle();
        }
        try {
            decodeFile = BitmapFactory.decodeFile(str, options);
            z = false;
        } catch (Exception unused) {
            Runtime.getRuntime().gc();
        } catch (OutOfMemoryError unused2) {
            if (decodeFile != null && !decodeFile.isRecycled()) {
                decodeFile.recycle();
            }
            Runtime.getRuntime().gc();
        }
        if (z) {
            try {
                decodeFile = BitmapFactory.decodeFile(str, options);
            } catch (Exception unused3) {
                options.inPreferredConfig = Bitmap.Config.RGB_565;
                decodeFile = BitmapFactory.decodeFile(str, options);
            }
        }
        if (decodeFile == null) {
            return decodeFile;
        }
        if (i == 90) {
            i += 180;
        }
        Bitmap rotateBitmapByDegree = rotateBitmapByDegree(decodeFile, i);
        return rotateBitmapByDegree.getWidth() >= 1080 ? zoomBitmap(rotateBitmapByDegree, 1080, (rotateBitmapByDegree.getHeight() * 1080) / rotateBitmapByDegree.getWidth()) : rotateBitmapByDegree;
    }

    public static String getImageTypeWithMime(String str) {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(str, options);
        String str2 = options.outMimeType;
        return TextUtils.isEmpty(str2) ? "" : str2.substring(6);
    }

    public static float getLongImageMaxScale(Context context, String str) {
        return getLongImageMinScale(context, str) * 2.0f;
    }

    public static float getLongImageMinScale(Context context, String str) {
        return PhoneUtil.getPhoneWid(context.getApplicationContext()) / getWidthHeight(str)[0];
    }

    public static int getOrientation(String str) {
        try {
            int attributeInt = new ExifInterface(str).getAttributeInt(androidx.exifinterface.media.ExifInterface.TAG_ORIENTATION, 1);
            if (attributeInt == 3) {
                return 180;
            }
            if (attributeInt != 6) {
                return attributeInt != 8 ? 0 : 270;
            }
            return 90;
        } catch (Exception e) {
            e.printStackTrace();
            return 0;
        }
    }

    public static float getSmallImageMaxScale(Context context, String str) {
        return getSmallImageMinScale(context, str) * 2.0f;
    }

    public static float getSmallImageMinScale(Context context, String str) {
        return PhoneUtil.getPhoneWid(context.getApplicationContext()) / getWidthHeight(str)[0];
    }

    public static float getWideImageDoubleScale(Context context, String str) {
        return PhoneUtil.getPhoneHei(context.getApplicationContext()) / getWidthHeight(str)[1];
    }

    public static int[] getWidthHeight(String str) {
        Bitmap decodeFile;
        if (str.isEmpty()) {
            return new int[]{0, 0};
        }
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        try {
            BitmapFactory.decodeFile(str, options);
        } catch (Exception e) {
            e.printStackTrace();
        }
        int i = options.outWidth;
        int i2 = options.outHeight;
        if (i2 == -1 || i == -1) {
            try {
                ExifInterface exifInterface = new ExifInterface(str);
                i2 = exifInterface.getAttributeInt(androidx.exifinterface.media.ExifInterface.TAG_IMAGE_LENGTH, 1);
                i = exifInterface.getAttributeInt(androidx.exifinterface.media.ExifInterface.TAG_IMAGE_WIDTH, 1);
            } catch (IOException e2) {
                e2.printStackTrace();
            }
        }
        if ((i <= 0 || i2 <= 0) && (decodeFile = BitmapFactory.decodeFile(str)) != null) {
            i = decodeFile.getWidth();
            i2 = decodeFile.getHeight();
            try {
                if (!decodeFile.isRecycled()) {
                    decodeFile.recycle();
                }
            } catch (Exception e3) {
                e3.printStackTrace();
            }
        }
        int orientation = getOrientation(str);
        return (orientation == 90 || orientation == 270) ? new int[]{i2, i} : new int[]{i, i2};
    }

    public static boolean isBmpImageWithMime(String str) {
        return "bmp".equalsIgnoreCase(getImageTypeWithMime(str));
    }

    public static boolean isGifImageWithMime(String str) {
        return "gif".equalsIgnoreCase(getImageTypeWithMime(str));
    }

    public static boolean isGifImageWithUrl(String str) {
        return str.toLowerCase().endsWith("gif");
    }

    public static boolean isLongImage(Context context, String str) {
        int[] widthHeight = getWidthHeight(str);
        boolean z = false;
        float f = widthHeight[0];
        float f2 = widthHeight[1];
        float f3 = f2 / f;
        float phoneRatio = PhoneUtil.getPhoneRatio(context.getApplicationContext()) + 0.1f;
        if (f > 0.0f && f2 > 0.0f && f2 > f && f3 >= phoneRatio) {
            z = true;
        }
        Print.d(TAG, "isLongImage = " + z);
        return z;
    }

    public static boolean isSmallImage(Context context, String str) {
        boolean z = getWidthHeight(str)[0] < PhoneUtil.getPhoneWid(context.getApplicationContext());
        Print.d(TAG, "isSmallImage = " + z);
        return z;
    }

    public static boolean isWideImage(Context context, String str) {
        int[] widthHeight = getWidthHeight(str);
        boolean z = false;
        float f = widthHeight[0];
        float f2 = widthHeight[1];
        float f3 = f / f2;
        if (f > 0.0f && f2 > 0.0f && f > f2 && f3 >= 2.0f) {
            z = true;
        }
        Print.d(TAG, "isWideImage = " + z);
        return z;
    }

    public static Bitmap resizeImage(Bitmap bitmap, int i, int i2) {
        if (bitmap == null) {
            return null;
        }
        int height = bitmap.getHeight();
        int width = bitmap.getWidth();
        Matrix matrix = new Matrix();
        matrix.postScale(i / width, i2 / height);
        Bitmap createBitmap = Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, false);
        if (!bitmap.isRecycled()) {
            bitmap.recycle();
        }
        return createBitmap;
    }

    /* JADX WARN: Removed duplicated region for block: B:10:0x0028  */
    /* JADX WARN: Removed duplicated region for block: B:8:0x0025  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static Bitmap rotateBitmapByDegree(Bitmap bitmap, int i) {
        Bitmap bitmap2;
        Bitmap bitmap3;
        Matrix matrix = new Matrix();
        matrix.postRotate(i);
        try {
            bitmap2 = bitmap;
            try {
                bitmap3 = Bitmap.createBitmap(bitmap2, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
            } catch (OutOfMemoryError e) {
                e = e;
                e.printStackTrace();
                bitmap3 = null;
                if (bitmap3 == null) {
                }
                if (bitmap2 != bitmap3) {
                }
                return bitmap3;
            }
        } catch (OutOfMemoryError e2) {
            e = e2;
            bitmap2 = bitmap;
        }
        if (bitmap3 == null) {
            bitmap3 = bitmap2;
        }
        if (bitmap2 != bitmap3) {
            bitmap2.recycle();
        }
        return bitmap3;
    }

    public static String saveBitmapBackPath(Bitmap bitmap) throws IOException {
        String str = Environment.getExternalStorageDirectory() + "/ShareLongPicture/.temp/";
        File file = new File(str);
        if (!file.exists()) {
            try {
                file.mkdirs();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        File file2 = new File(str + ("temp_LongPictureShare_" + System.currentTimeMillis() + PictureMimeType.JPEG));
        if (!file2.exists()) {
            file2.createNewFile();
        }
        BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(new FileOutputStream(file2));
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, bufferedOutputStream);
        bufferedOutputStream.flush();
        bufferedOutputStream.close();
        return file2.getAbsolutePath();
    }

    private static Bitmap zoomBitmap(Bitmap bitmap, int i, int i2) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        Matrix matrix = new Matrix();
        matrix.postScale(i / width, i2 / height);
        return Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true);
    }
}
