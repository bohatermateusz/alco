package com.hismith.hismithapp.image_preview.glide.progress;

import android.text.TextUtils;
import com.hismith.hismithapp.image_preview.glide.SSLSocketClient;
import com.hismith.hismithapp.image_preview.glide.progress.ProgressResponseBody;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import okhttp3.Interceptor;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

/* loaded from: classes3.dex */
public class ProgressManager {
    private static Map<String, OnProgressListener> listenersMap = Collections.synchronizedMap(new HashMap());
    private static final ProgressResponseBody.InternalProgressListener LISTENER = new ProgressResponseBody.InternalProgressListener() { // from class: com.hismith.hismithapp.image_preview.glide.progress.ProgressManager.1
        @Override // com.hismith.hismithapp.image_preview.glide.progress.ProgressResponseBody.InternalProgressListener
        public void onProgress(String str, long j, long j2) {
            OnProgressListener progressListener = ProgressManager.getProgressListener(str);
            if (progressListener != null) {
                int i = (int) (((j * 1.0f) / j2) * 100.0f);
                boolean z = i >= 100;
                progressListener.onProgress(str, z, i, j, j2);
                if (z) {
                    ProgressManager.removeListener(str);
                }
            }
        }
    };

    private ProgressManager() {
    }

    public static void addListener(String str, OnProgressListener onProgressListener) {
        if (TextUtils.isEmpty(str) || onProgressListener == null) {
            return;
        }
        listenersMap.put(str, onProgressListener);
        onProgressListener.onProgress(str, false, 1, 0L, 0L);
    }

    public static OkHttpClient getOkHttpClient() {
        OkHttpClient.Builder builder = new OkHttpClient.Builder();
        builder.addNetworkInterceptor(new Interceptor() { // from class: com.hismith.hismithapp.image_preview.glide.progress.ProgressManager.2
            @Override // okhttp3.Interceptor
            public Response intercept(Interceptor.Chain chain) throws IOException {
                Request request = chain.request();
                Response proceed = chain.proceed(request);
                return proceed.newBuilder().body(new ProgressResponseBody(request.url().toString(), ProgressManager.LISTENER, proceed.body())).build();
            }
        }).sslSocketFactory(SSLSocketClient.getSSLSocketFactory(), SSLSocketClient.geX509tTrustManager()).hostnameVerifier(SSLSocketClient.getHostnameVerifier());
        builder.connectTimeout(30L, TimeUnit.SECONDS);
        builder.writeTimeout(30L, TimeUnit.SECONDS);
        builder.readTimeout(30L, TimeUnit.SECONDS);
        return builder.build();
    }

    public static OnProgressListener getProgressListener(String str) {
        Map<String, OnProgressListener> map;
        if (TextUtils.isEmpty(str) || (map = listenersMap) == null || map.size() == 0) {
            return null;
        }
        return listenersMap.get(str);
    }

    public static void removeListener(String str) {
        if (TextUtils.isEmpty(str)) {
            return;
        }
        listenersMap.remove(str);
    }
}
