package com.hismith.hismithapp.image_preview.glide.progress;

import android.os.Handler;
import android.os.Looper;
import java.io.IOException;
import okhttp3.MediaType;
import okhttp3.ResponseBody;
import okio.Buffer;
import okio.BufferedSource;
import okio.ForwardingSource;
import okio.Okio;
import okio.Source;

/* loaded from: classes3.dex */
public class ProgressResponseBody extends ResponseBody {
    private static final Handler mainThreadHandler = new Handler(Looper.getMainLooper());
    private BufferedSource bufferedSource;
    private InternalProgressListener internalProgressListener;
    private ResponseBody responseBody;
    private String url;

    /* renamed from: com.hismith.hismithapp.image_preview.glide.progress.ProgressResponseBody$1, reason: invalid class name */
    class AnonymousClass1 extends ForwardingSource {
        long lastTotalBytesRead;
        long totalBytesRead;

        AnonymousClass1(Source source) {
            super(source);
        }

        @Override // okio.ForwardingSource, okio.Source
        public long read(Buffer buffer, long j) throws IOException {
            long read = super.read(buffer, j);
            this.totalBytesRead += read == -1 ? 0L : read;
            if (ProgressResponseBody.this.internalProgressListener != null) {
                long j2 = this.lastTotalBytesRead;
                long j3 = this.totalBytesRead;
                if (j2 != j3) {
                    this.lastTotalBytesRead = j3;
                    ProgressResponseBody.mainThreadHandler.post(new Runnable() { // from class: com.hismith.hismithapp.image_preview.glide.progress.ProgressResponseBody.1.1
                        @Override // java.lang.Runnable
                        public void run() {
                            ProgressResponseBody.this.internalProgressListener.onProgress(ProgressResponseBody.this.url, AnonymousClass1.this.totalBytesRead, ProgressResponseBody.this.contentLength());
                        }
                    });
                }
            }
            return read;
        }
    }

    interface InternalProgressListener {
        void onProgress(String str, long j, long j2);
    }

    ProgressResponseBody(String str, InternalProgressListener internalProgressListener, ResponseBody responseBody) {
        this.url = str;
        this.internalProgressListener = internalProgressListener;
        this.responseBody = responseBody;
    }

    private Source source(Source source) {
        return new AnonymousClass1(source);
    }

    @Override // okhttp3.ResponseBody
    public long contentLength() {
        return this.responseBody.contentLength();
    }

    @Override // okhttp3.ResponseBody
    public MediaType contentType() {
        return this.responseBody.contentType();
    }

    @Override // okhttp3.ResponseBody
    public BufferedSource source() {
        BufferedSource bufferedSource = this.bufferedSource;
        if (bufferedSource != null) {
            return bufferedSource;
        }
        BufferedSource buffer = Okio.buffer(source(this.responseBody.source()));
        this.bufferedSource = buffer;
        return buffer;
    }
}
