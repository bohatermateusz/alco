package com.hismith.hismithapp.utils;

import android.content.ContentUris;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.util.Log;
import com.google.firebase.analytics.FirebaseAnalytics;
import com.luck.picture.lib.config.PictureMimeType;
import com.twitter.sdk.android.core.internal.scribe.EventsFilesManager;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.UUID;

/* loaded from: classes3.dex */
public final class UriUtils {
    private UriUtils() {
        throw new AssertionError();
    }

    public static String generateRandomFileName(String str) {
        if (str != null && !str.startsWith(".")) {
            str = "." + str;
        }
        StringBuilder append = new StringBuilder("file_").append(new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date())).append(EventsFilesManager.ROLL_OVER_FILE_NAME_SEPARATOR).append(UUID.randomUUID().toString().substring(0, 8));
        if (str == null) {
            str = PictureMimeType.JPG;
        }
        return append.append(str).toString();
    }

    public static String getExtensionFromMimeType(String str) {
        if (str == null) {
            return PictureMimeType.JPG;
        }
        str.hashCode();
        switch (str) {
        }
        return PictureMimeType.JPG;
    }

    public static String getFileExtensionFromUri(Context context, Uri uri) {
        String fileNameFromUri = getFileNameFromUri(context, uri);
        return (fileNameFromUri == null || !fileNameFromUri.contains(".")) ? getExtensionFromMimeType(context.getContentResolver().getType(uri)) : fileNameFromUri.substring(fileNameFromUri.lastIndexOf("."));
    }

    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Type inference failed for: r6v0, types: [android.content.Context] */
    /* JADX WARN: Type inference failed for: r6v2 */
    /* JADX WARN: Type inference failed for: r6v3 */
    /* JADX WARN: Type inference failed for: r6v4, types: [java.io.InputStream] */
    /* JADX WARN: Type inference failed for: r6v6, types: [java.io.InputStream] */
    /* JADX WARN: Type inference failed for: r6v9, types: [java.io.InputStream] */
    /* JADX WARN: Type inference failed for: r7v0, types: [android.net.Uri] */
    /* JADX WARN: Type inference failed for: r7v1 */
    /* JADX WARN: Type inference failed for: r7v3, types: [java.io.OutputStream] */
    /* JADX WARN: Type inference failed for: r7v6 */
    public static String getFileFromUri(Context context, Uri uri) {
        Throwable th;
        FileOutputStream fileOutputStream;
        File file = new File(context.getCacheDir(), generateRandomFileName(getFileExtensionFromUri(context, uri)));
        try {
            try {
                context = context.getContentResolver().openInputStream(uri);
                if (context == 0) {
                    if (context != 0) {
                        try {
                            context.close();
                            return null;
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                    return null;
                }
                try {
                    fileOutputStream = new FileOutputStream(file);
                    try {
                        byte[] bArr = new byte[4096];
                        while (true) {
                            int read = context.read(bArr);
                            if (read == -1) {
                                break;
                            }
                            fileOutputStream.write(bArr, 0, read);
                        }
                        fileOutputStream.flush();
                        String path = file.getPath();
                        if (context != 0) {
                            try {
                                context.close();
                            } catch (IOException e2) {
                                e2.printStackTrace();
                                return path;
                            }
                        }
                        fileOutputStream.close();
                        return path;
                    } catch (IOException e3) {
                        e = e3;
                        e.printStackTrace();
                        if (file.exists()) {
                            file.delete();
                        }
                        if (context != 0) {
                            try {
                                context.close();
                            } catch (IOException e4) {
                                e4.printStackTrace();
                                return null;
                            }
                        }
                        if (fileOutputStream != null) {
                            fileOutputStream.close();
                        }
                        return null;
                    }
                } catch (IOException e5) {
                    e = e5;
                    fileOutputStream = null;
                } catch (Throwable th2) {
                    th = th2;
                    uri = 0;
                    if (context != 0) {
                        try {
                            context.close();
                        } catch (IOException e6) {
                            e6.printStackTrace();
                            throw th;
                        }
                    }
                    if (uri != 0) {
                        uri.close();
                    }
                    throw th;
                }
            } catch (Throwable th3) {
                th = th3;
            }
        } catch (IOException e7) {
            e = e7;
            context = 0;
            fileOutputStream = null;
        } catch (Throwable th4) {
            uri = 0;
            th = th4;
            context = 0;
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:14:? A[RETURN, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:5:0x0051  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static String getFileNameFromUri(Context context, Uri uri) {
        Uri uri2;
        int columnIndex;
        String str = null;
        if (uri.getScheme().equals(FirebaseAnalytics.Param.CONTENT)) {
            try {
                uri2 = uri;
            } catch (Exception e) {
                e = e;
                uri2 = uri;
            }
            try {
                Cursor query = context.getContentResolver().query(uri2, null, null, null, null);
                if (query != null) {
                    try {
                        if (query.moveToFirst() && (columnIndex = query.getColumnIndex("_display_name")) != -1) {
                            str = query.getString(columnIndex);
                        }
                    } finally {
                    }
                }
                if (query != null) {
                    query.close();
                }
            } catch (Exception e2) {
                e = e2;
                e.printStackTrace();
                if (str == null) {
                }
            }
        } else {
            uri2 = uri;
        }
        if (str == null) {
            return str;
        }
        String path = uri2.getPath();
        int lastIndexOf = path != null ? path.lastIndexOf(47) : -1;
        return lastIndexOf != -1 ? path.substring(lastIndexOf + 1) : path;
    }

    public static String getImagePath(Context context, Intent intent) {
        Uri data = intent.getData();
        LogUtil.e("uri=intent.getData :" + data);
        Log.d("uri=intent.getData :", "" + data);
        if (DocumentsContract.isDocumentUri(context, data)) {
            String documentId = DocumentsContract.getDocumentId(data);
            LogUtil.e("getDocumentId(uri) :" + documentId);
            Log.d("getDocumentId(uri) :", "" + documentId);
            LogUtil.e("uri.getAuthority() :" + data.getAuthority());
            Log.d("uri.getAuthority() :", "" + data.getAuthority());
            if ("com.android.providers.media.documents".equals(data.getAuthority())) {
                return getImagePath(context, MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "_id=" + documentId.split(":")[1]);
            }
            if ("com.android.providers.downloads.documents".equals(data.getAuthority())) {
                return getImagePath(context, ContentUris.withAppendedId(Uri.parse("content://downloads/public_downloads"), Long.valueOf(documentId).longValue()), null);
            }
        } else if (FirebaseAnalytics.Param.CONTENT.equalsIgnoreCase(data.getScheme())) {
            return "com.google.android.apps.photos.contentprovider".equals(data.getAuthority()) ? getFileFromUri(context, data) : getImagePath(context, data, null);
        }
        return null;
    }

    private static String getImagePath(Context context, Uri uri, String str) {
        Cursor query = context.getContentResolver().query(uri, null, str, null, null);
        if (query != null) {
            r7 = query.moveToFirst() ? query.getString(query.getColumnIndex("_data")) : null;
            query.close();
        }
        return r7;
    }
}
