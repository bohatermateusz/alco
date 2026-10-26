package com.hismith.hismithapp.utils;

import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.util.Log;
import com.bumptech.glide.load.Key;
import com.luck.picture.lib.config.SelectMimeType;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URLEncoder;
import java.util.Map;
import okhttp3.MediaType;
import okhttp3.RequestBody;

/* loaded from: classes3.dex */
public class RequestBodyUtil {
    public static void addMultiPart(Context context, Map<String, RequestBody> map, String str, Object obj) {
        if (obj instanceof String) {
            map.put(str, RequestBody.create(MediaType.parse("text/plain;charset=UTF-8"), (String) obj));
            return;
        }
        if (obj instanceof File) {
            try {
                String encode = URLEncoder.encode(((File) obj).getName(), Key.STRING_CHARSET_NAME);
                Log.i("UploadFile", "getRequestParams: " + encode);
                map.put(str + "\"; filename= \"" + encode + "", RequestBody.create(MediaType.parse("application/octet-stream"), (File) obj));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public static void addMultiPartUri(Context context, Map<String, RequestBody> map, String str, Uri uri) {
        if (uri != null) {
            try {
                InputStream openInputStream = context.getContentResolver().openInputStream(uri);
                if (openInputStream == null) {
                    return;
                }
                map.put("file\"; filename=\"" + str, RequestBody.create(MediaType.parse(SelectMimeType.SYSTEM_IMAGE), readBytes(openInputStream)));
                openInputStream.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public static String getFileNameFromUri(Context context, Uri uri) {
        Cursor query = context.getContentResolver().query(uri, null, null, null, null);
        if (query == null) {
            return null;
        }
        int columnIndex = query.getColumnIndex("_display_name");
        if (columnIndex < 0 || !query.moveToFirst()) {
            query.close();
            return null;
        }
        String string = query.getString(columnIndex);
        query.close();
        return string;
    }

    private static byte[] readBytes(InputStream inputStream) throws IOException {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        byte[] bArr = new byte[1024];
        while (true) {
            int read = inputStream.read(bArr, 0, 1024);
            if (read == -1) {
                byteArrayOutputStream.flush();
                return byteArrayOutputStream.toByteArray();
            }
            byteArrayOutputStream.write(bArr, 0, read);
        }
    }
}
