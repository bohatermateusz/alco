package com.hismith.hismithapp.utils;

import android.app.Activity;
import android.content.ContentUris;
import android.content.Context;
import android.content.res.AssetManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Environment;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;
import com.google.firebase.analytics.FirebaseAnalytics;
import com.hismith.hismithapp.application.MyApplication;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

/* loaded from: classes3.dex */
public class FileUtil {
    private static final String TAG = "FileUtil";
    private static File cacheDir = MyApplication.getApplication().getCacheDir();
    private static String pathDiv = "/";

    public enum FileType {
        IMG,
        AUDIO,
        VIDEO,
        FILE
    }

    private FileUtil() {
        throw new UnsupportedOperationException("cannot be instantiated");
    }

    public static String createFile(Bitmap bitmap, String str) {
        File file = new File(cacheDir, str);
        try {
            if (file.createNewFile()) {
                ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
                bitmap.compress(Bitmap.CompressFormat.PNG, 0, byteArrayOutputStream);
                byte[] byteArray = byteArrayOutputStream.toByteArray();
                FileOutputStream fileOutputStream = new FileOutputStream(file);
                fileOutputStream.write(byteArray);
                fileOutputStream.flush();
                fileOutputStream.close();
            }
        } catch (IOException e) {
            Log.e(TAG, "create bitmap file error" + e);
        }
        if (file.exists()) {
            return file.getAbsolutePath();
        }
        return null;
    }

    public static void createFile(byte[] bArr, String str) {
        File file = new File(cacheDir, str);
        try {
            if (file.createNewFile()) {
                FileOutputStream fileOutputStream = new FileOutputStream(file);
                fileOutputStream.write(bArr);
                fileOutputStream.flush();
                fileOutputStream.close();
            }
        } catch (IOException e) {
            Log.e(TAG, "create file error" + e);
        }
    }

    public static boolean createFile(byte[] bArr, String str, String str2) {
        File externalFilesDir;
        if (isExternalStorageWritable() && (externalFilesDir = MyApplication.getApplication().getExternalFilesDir(str2)) != null) {
            File file = new File(externalFilesDir, str);
            try {
                if (file.createNewFile()) {
                    FileOutputStream fileOutputStream = new FileOutputStream(file);
                    fileOutputStream.write(bArr);
                    fileOutputStream.flush();
                    fileOutputStream.close();
                    return true;
                }
            } catch (IOException e) {
                Log.e(TAG, "create file error" + e);
            }
        }
        return false;
    }

    public static String getCacheFilePath(String str) {
        return cacheDir.getAbsolutePath() + pathDiv + str;
    }

    private static String getDataColumn(Context context, Uri uri, String str, String[] strArr) {
        Throwable th;
        Cursor cursor = null;
        try {
            Cursor query = context.getContentResolver().query(uri, new String[]{"_data"}, str, strArr, null);
            if (query != null) {
                try {
                    if (query.moveToFirst()) {
                        String string = query.getString(query.getColumnIndexOrThrow("_data"));
                        if (query != null) {
                            query.close();
                        }
                        return string;
                    }
                } catch (Throwable th2) {
                    th = th2;
                    cursor = query;
                    if (cursor == null) {
                        throw th;
                    }
                    cursor.close();
                    throw th;
                }
            }
            if (query != null) {
                query.close();
            }
            return null;
        } catch (Throwable th3) {
            th = th3;
        }
    }

    public static String getFilePath(Context context, Uri uri) {
        Uri uri2 = null;
        if (DocumentsContract.isDocumentUri(context, uri)) {
            if (isExternalStorageDocument(uri)) {
                String[] split = DocumentsContract.getDocumentId(uri).split(":");
                if ("primary".equalsIgnoreCase(split[0])) {
                    return Environment.getExternalStorageDirectory() + "/" + split[1];
                }
            } else {
                if (isDownloadsDocument(uri)) {
                    return getDataColumn(context, ContentUris.withAppendedId(Uri.parse("content://downloads/public_downloads"), Long.valueOf(DocumentsContract.getDocumentId(uri)).longValue()), null, null);
                }
                if (isMediaDocument(uri)) {
                    String[] split2 = DocumentsContract.getDocumentId(uri).split(":");
                    String str = split2[0];
                    if ("image".equals(str)) {
                        uri2 = MediaStore.Images.Media.EXTERNAL_CONTENT_URI;
                    } else if ("video".equals(str)) {
                        uri2 = MediaStore.Video.Media.EXTERNAL_CONTENT_URI;
                    } else if ("audio".equals(str)) {
                        uri2 = MediaStore.Audio.Media.EXTERNAL_CONTENT_URI;
                    }
                    return getDataColumn(context, uri2, "_id=?", new String[]{split2[1]});
                }
            }
        } else {
            if (FirebaseAnalytics.Param.CONTENT.equalsIgnoreCase(uri.getScheme())) {
                return getDataColumn(context, uri, null, null);
            }
            if ("file".equalsIgnoreCase(uri.getScheme())) {
                return uri.getPath();
            }
        }
        return null;
    }

    public static long getFileSize(File file) {
        FileInputStream fileInputStream;
        FileInputStream fileInputStream2 = null;
        try {
            try {
                fileInputStream = new FileInputStream(file);
            } catch (Exception e) {
                e = e;
            }
        } catch (Throwable th) {
            th = th;
        }
        try {
            long available = fileInputStream.available();
            try {
                fileInputStream.close();
                return available;
            } catch (IOException e2) {
                e2.printStackTrace();
                return available;
            }
        } catch (Exception e3) {
            e = e3;
            fileInputStream2 = fileInputStream;
            e.printStackTrace();
            if (fileInputStream2 != null) {
                try {
                    fileInputStream2.close();
                } catch (IOException e4) {
                    e4.printStackTrace();
                }
            }
            return 0L;
        } catch (Throwable th2) {
            th = th2;
            fileInputStream2 = fileInputStream;
            if (fileInputStream2 != null) {
                try {
                    fileInputStream2.close();
                } catch (IOException e5) {
                    e5.printStackTrace();
                }
            }
            throw th;
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:14:0x0047  */
    /* JADX WARN: Removed duplicated region for block: B:9:0x0028  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static String getImageFilePath(Context context, Uri uri) {
        String dataColumn;
        if (uri == null) {
            return null;
        }
        if (isMediaDocument(uri)) {
            try {
                dataColumn = getDataColumn(context, MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "_id=?", new String[]{DocumentsContract.getDocumentId(uri).split(":")[1]});
            } catch (IllegalArgumentException unused) {
            }
            if (dataColumn == null) {
                return dataColumn;
            }
            Cursor managedQuery = ((Activity) context).managedQuery(uri, new String[]{"_data"}, null, null, null);
            if (managedQuery == null) {
                return null;
            }
            int columnIndexOrThrow = managedQuery.getColumnIndexOrThrow("_data");
            managedQuery.moveToFirst();
            return managedQuery.getString(columnIndexOrThrow);
        }
        dataColumn = null;
        if (dataColumn == null) {
        }
    }

    public static File getTempFile(FileType fileType) {
        try {
            File createTempFile = File.createTempFile(fileType.toString(), null, cacheDir);
            createTempFile.deleteOnExit();
            return createTempFile;
        } catch (IOException unused) {
            return null;
        }
    }

    public static boolean isCacheFileExist(String str) {
        return new File(getCacheFilePath(str)).exists();
    }

    public static boolean isDownloadsDocument(Uri uri) {
        return "com.android.providers.downloads.documents".equals(uri.getAuthority());
    }

    public static boolean isExternalStorageDocument(Uri uri) {
        return "com.android.externalstorage.documents".equals(uri.getAuthority());
    }

    public static boolean isExternalStorageWritable() {
        if ("mounted".equals(Environment.getExternalStorageState())) {
            return true;
        }
        Log.e(TAG, "ExternalStorage not mounted");
        return false;
    }

    public static boolean isMediaDocument(Uri uri) {
        return "com.android.providers.media.documents".equals(uri.getAuthority());
    }

    public static void readAssetsImageBase64Data(Context context) {
        try {
            AssetManager assets = context.getAssets();
            String substring = "文件名.png".substring("文件名.png".lastIndexOf(".") + 1);
            InputStream open = assets.open("文件名.png");
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            byte[] bArr = new byte[1024];
            while (true) {
                int read = open.read(bArr);
                if (read == -1) {
                    byteArrayOutputStream.flush();
                    LogUtil.e("图片数据：" + ("data:image/" + substring + ";base64," + Base64.encodeToString(byteArrayOutputStream.toByteArray(), 0)));
                    return;
                }
                byteArrayOutputStream.write(bArr, 0, read);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
