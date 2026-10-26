package com.hismith.hismithapp.utils;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;
import com.luck.picture.lib.config.PictureMimeType;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;

/* loaded from: classes3.dex */
public class ImageCompressor {
    public static String compressImage(Context context, String str, int i, int i2) {
        Exception exc;
        Bitmap bitmap;
        try {
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inJustDecodeBounds = true;
            InputStream inputStream = getInputStream(context, str);
            if (inputStream == null) {
                return null;
            }
            BitmapFactory.decodeStream(inputStream, null, options);
            inputStream.close();
            int i3 = options.outWidth;
            int i4 = options.outHeight;
            if (i3 > 0 && i4 > 0) {
                options.inSampleSize = i3 > i ? Math.round(i3 / i) : 1;
                options.inJustDecodeBounds = false;
                InputStream inputStream2 = getInputStream(context, str);
                bitmap = BitmapFactory.decodeStream(inputStream2, null, options);
                try {
                    inputStream2.close();
                    if (bitmap == null) {
                        return null;
                    }
                    int rotationFromUri = getRotationFromUri(context, str);
                    if (rotationFromUri != 0) {
                        Matrix matrix = new Matrix();
                        matrix.postRotate(rotationFromUri);
                        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
                    }
                    ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
                    int i5 = 90;
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 90, byteArrayOutputStream);
                    while (byteArrayOutputStream.toByteArray().length / 1024 > i2 && i5 > 20) {
                        byteArrayOutputStream.reset();
                        i5 -= 10;
                        bitmap.compress(Bitmap.CompressFormat.JPEG, i5, byteArrayOutputStream);
                    }
                    File file = new File(context.getCacheDir(), "upload_img_" + System.currentTimeMillis() + PictureMimeType.JPG);
                    FileOutputStream fileOutputStream = new FileOutputStream(file);
                    fileOutputStream.write(byteArrayOutputStream.toByteArray());
                    fileOutputStream.close();
                    bitmap.recycle();
                    return file.getAbsolutePath();
                } catch (Exception e) {
                    exc = e;
                    exc.printStackTrace();
                    if (bitmap != null) {
                        bitmap.recycle();
                    }
                    return null;
                }
            }
            return null;
        } catch (Exception e2) {
            exc = e2;
            bitmap = null;
        }
    }

    private static InputStream getInputStream(Context context, String str) throws Exception {
        return str.startsWith("content://") ? context.getContentResolver().openInputStream(Uri.parse(str)) : new FileInputStream(str);
    }

    private static int getRotationFromUri(Context context, String str) {
        ExifInterface exifInterface;
        try {
            if (str.startsWith("content://")) {
                AssetFileDescriptor openAssetFileDescriptor = context.getContentResolver().openAssetFileDescriptor(Uri.parse(str), "r");
                if (openAssetFileDescriptor == null) {
                    if (openAssetFileDescriptor != null) {
                        openAssetFileDescriptor.close();
                    }
                    return 0;
                }
                try {
                    exifInterface = new ExifInterface(openAssetFileDescriptor.getFileDescriptor());
                    if (openAssetFileDescriptor != null) {
                        openAssetFileDescriptor.close();
                    }
                } finally {
                }
            } else {
                exifInterface = new ExifInterface(str);
            }
            int attributeInt = exifInterface.getAttributeInt(androidx.exifinterface.media.ExifInterface.TAG_ORIENTATION, 1);
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
}
