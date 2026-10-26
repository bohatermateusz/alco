package com.hismith.hismithapp.utils;

import android.content.ContentUris;
import android.content.Context;
import android.content.Intent;
import android.content.res.Resources;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Environment;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.text.TextUtils;
import com.google.firebase.analytics.FirebaseAnalytics;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

/* loaded from: classes3.dex */
public class ImageUtil {
    public static Bitmap compressImage(Bitmap bitmap) {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
        int i = 100;
        while (byteArrayOutputStream.toByteArray().length / 1024 > 100) {
            byteArrayOutputStream.reset();
            bitmap.compress(Bitmap.CompressFormat.JPEG, i, byteArrayOutputStream);
            i -= 10;
            if (i <= 0) {
                break;
            }
        }
        return BitmapFactory.decodeStream(new ByteArrayInputStream(byteArrayOutputStream.toByteArray()), null, null);
    }

    public static File compressImageFile(String str, Bitmap bitmap) {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
        int i = 100;
        while (byteArrayOutputStream.toByteArray().length / 1024 > 100) {
            byteArrayOutputStream.reset();
            i -= 10;
            bitmap.compress(Bitmap.CompressFormat.JPEG, i, byteArrayOutputStream);
        }
        File file = new File(Environment.getExternalStorageDirectory() + "/" + str);
        try {
            FileOutputStream fileOutputStream = new FileOutputStream(file);
            try {
                fileOutputStream.write(byteArrayOutputStream.toByteArray());
                fileOutputStream.flush();
                fileOutputStream.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        } catch (FileNotFoundException e2) {
            e2.printStackTrace();
        }
        return file;
    }

    public static Bitmap compressImageFromFile(String str, float f) {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(str, options);
        options.inJustDecodeBounds = false;
        int i = options.outWidth;
        int i2 = options.outHeight;
        float f2 = i2;
        float f3 = i;
        float f4 = (f * f2) / f3;
        int i3 = (i <= i2 || f3 <= f) ? (i >= i2 || f2 <= f4) ? 1 : (int) (options.outHeight / f4) : (int) (options.outWidth / f);
        if (i3 <= 0) {
            i3 = 1;
        }
        options.inSampleSize = i3;
        options.inPurgeable = true;
        options.inInputShareable = true;
        return BitmapFactory.decodeFile(str, options);
    }

    /* JADX WARN: Removed duplicated region for block: B:9:0x0042  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static Bitmap getBitmapFormUri(Context context, Uri uri) throws FileNotFoundException, IOException {
        int i;
        InputStream openInputStream = context.getContentResolver().openInputStream(uri);
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        options.inDither = true;
        options.inPreferredConfig = Bitmap.Config.RGB_565;
        BitmapFactory.decodeStream(openInputStream, null, options);
        openInputStream.close();
        int i2 = options.outWidth;
        int i3 = options.outHeight;
        if (i2 == -1 || i3 == -1) {
            return null;
        }
        if (i2 > i3) {
            float f = i2;
            if (f > 480.0f) {
                i = (int) (f / 480.0f);
                if (i <= 0) {
                    i = 1;
                }
                BitmapFactory.Options options2 = new BitmapFactory.Options();
                options2.inSampleSize = i;
                options2.inDither = true;
                options2.inPreferredConfig = Bitmap.Config.RGB_565;
                InputStream openInputStream2 = context.getContentResolver().openInputStream(uri);
                Bitmap decodeStream = BitmapFactory.decodeStream(openInputStream2, null, options2);
                openInputStream2.close();
                return compressImage(decodeStream);
            }
        }
        if (i2 < i3) {
            float f2 = i3;
            if (f2 > 800.0f) {
                i = (int) (f2 / 800.0f);
                if (i <= 0) {
                }
                BitmapFactory.Options options22 = new BitmapFactory.Options();
                options22.inSampleSize = i;
                options22.inDither = true;
                options22.inPreferredConfig = Bitmap.Config.RGB_565;
                InputStream openInputStream22 = context.getContentResolver().openInputStream(uri);
                Bitmap decodeStream2 = BitmapFactory.decodeStream(openInputStream22, null, options22);
                openInputStream22.close();
                return compressImage(decodeStream2);
            }
        }
        i = 1;
        if (i <= 0) {
        }
        BitmapFactory.Options options222 = new BitmapFactory.Options();
        options222.inSampleSize = i;
        options222.inDither = true;
        options222.inPreferredConfig = Bitmap.Config.RGB_565;
        InputStream openInputStream222 = context.getContentResolver().openInputStream(uri);
        Bitmap decodeStream22 = BitmapFactory.decodeStream(openInputStream222, null, options222);
        openInputStream222.close();
        return compressImage(decodeStream22);
    }

    /* JADX WARN: Removed duplicated region for block: B:12:0x003b  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static Bitmap getImage(String str) {
        int i;
        float f;
        if (TextUtils.isEmpty(str)) {
            return null;
        }
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(str, options);
        options.inJustDecodeBounds = false;
        int i2 = options.outWidth;
        int i3 = options.outHeight;
        if (i2 > i3 && i2 > 480.0f) {
            f = options.outWidth / 480.0f;
        } else {
            if (i2 >= i3 || i3 <= 800.0f) {
                i = 1;
                options.inSampleSize = i > 0 ? i : 1;
                return compressImage(BitmapFactory.decodeFile(str, options));
            }
            f = options.outHeight / 800.0f;
        }
        i = (int) f;
        options.inSampleSize = i > 0 ? i : 1;
        return compressImage(BitmapFactory.decodeFile(str, options));
    }

    public static String getImagePath(Context context, Uri uri, String str) {
        Cursor query = context.getContentResolver().query(uri, null, str, null, null);
        if (query != null) {
            r7 = query.moveToFirst() ? query.getString(query.getColumnIndex("_data")) : null;
            query.close();
        }
        return r7;
    }

    public static Bitmap handleImageBeforeKitKat(Context context, Intent intent) {
        return getImage(getImagePath(context, intent.getData(), null));
    }

    public static Bitmap handleImageOnKitKat(Context context, Intent intent) {
        String imagePath;
        Uri data = intent.getData();
        String str = null;
        if (DocumentsContract.isDocumentUri(context, data)) {
            String documentId = DocumentsContract.getDocumentId(data);
            if ("com.android.providers.media.documents".equals(data.getAuthority())) {
                imagePath = getImagePath(context, MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "_id=" + documentId.split(":")[1]);
            } else if ("com.android.providers.downloads.documents".equals(data.getAuthority())) {
                imagePath = getImagePath(context, ContentUris.withAppendedId(Uri.parse("content://downloads/public_downloads"), Long.valueOf(documentId).longValue()), null);
            }
            str = imagePath;
        } else if (FirebaseAnalytics.Param.CONTENT.equalsIgnoreCase(data.getScheme())) {
            str = getImagePath(context, data, null);
        }
        return getImage(str);
    }

    public static String handleImagePathBeforeKitKat(Context context, Intent intent) {
        return getImagePath(context, intent.getData(), null);
    }

    public static String handleImagePathOnKitKat(Context context, Intent intent) {
        Uri data = intent.getData();
        if (DocumentsContract.isDocumentUri(context, data)) {
            String documentId = DocumentsContract.getDocumentId(data);
            if ("com.android.providers.media.documents".equals(data.getAuthority())) {
                return getImagePath(context, MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "_id=" + documentId.split(":")[1]);
            }
            if ("com.android.providers.downloads.documents".equals(data.getAuthority())) {
                return getImagePath(context, ContentUris.withAppendedId(Uri.parse("content://downloads/public_downloads"), Long.valueOf(documentId).longValue()), null);
            }
        } else if (FirebaseAnalytics.Param.CONTENT.equalsIgnoreCase(data.getScheme())) {
            return getImagePath(context, data, null);
        }
        return null;
    }

    public static Bitmap openImageBitmap(String str) {
        Bitmap bitmap = null;
        try {
            BufferedInputStream bufferedInputStream = new BufferedInputStream(new FileInputStream(str));
            bitmap = BitmapFactory.decodeStream(bufferedInputStream);
            bufferedInputStream.close();
            return bitmap;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return bitmap;
        } catch (IOException e2) {
            e2.printStackTrace();
            return bitmap;
        }
    }

    public static Drawable openImageDrawable(Resources resources, String str) {
        Bitmap openImageBitmap = openImageBitmap(str);
        if (openImageBitmap != null) {
            return new BitmapDrawable(resources, openImageBitmap);
        }
        return null;
    }
}
