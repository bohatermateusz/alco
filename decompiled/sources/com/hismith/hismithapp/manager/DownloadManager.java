package com.hismith.hismithapp.manager;

import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.utils.IOUtils;
import io.reactivex.Observable;
import io.reactivex.ObservableEmitter;
import io.reactivex.ObservableOnSubscribe;
import io.reactivex.ObservableSource;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.functions.Function;
import io.reactivex.functions.Predicate;
import io.reactivex.schedulers.Schedulers;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.concurrent.atomic.AtomicReference;
import okhttp3.Call;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

/* loaded from: classes3.dex */
public class DownloadManager {
    private static final AtomicReference<DownloadManager> INSTANCE = new AtomicReference<>();
    private HashMap<String, Call> downCalls = new HashMap<>();
    private OkHttpClient mClient = new OkHttpClient.Builder().build();

    private class DownloadSubscribe implements ObservableOnSubscribe<DownloadInfo> {
        private DownloadInfo downloadInfo;

        public DownloadSubscribe(DownloadInfo downloadInfo) {
            this.downloadInfo = downloadInfo;
        }

        @Override // io.reactivex.ObservableOnSubscribe
        public void subscribe(ObservableEmitter<DownloadInfo> observableEmitter) throws Exception {
            InputStream inputStream;
            FileOutputStream fileOutputStream;
            String url = this.downloadInfo.getUrl();
            long progress = this.downloadInfo.getProgress();
            long total = this.downloadInfo.getTotal();
            observableEmitter.onNext(this.downloadInfo);
            Call newCall = DownloadManager.this.mClient.newCall(new Request.Builder().addHeader("RANGE", "bytes=" + progress + "-" + total).url(url).build());
            DownloadManager.this.downCalls.put(url, newCall);
            Response execute = newCall.execute();
            File file = new File(MyApplication.getApplication().getFilesDir(), this.downloadInfo.getFileName());
            FileOutputStream fileOutputStream2 = null;
            try {
                inputStream = execute.body().byteStream();
                try {
                    fileOutputStream = new FileOutputStream(file, true);
                } catch (Throwable th) {
                    th = th;
                }
            } catch (Throwable th2) {
                th = th2;
                inputStream = null;
            }
            try {
                byte[] bArr = new byte[2048];
                while (true) {
                    int read = inputStream.read(bArr);
                    if (read == -1) {
                        fileOutputStream.flush();
                        DownloadManager.this.downCalls.remove(url);
                        IOUtils.closeQuietly(fileOutputStream);
                        IOUtils.closeQuietly(inputStream);
                        observableEmitter.onComplete();
                        return;
                    }
                    fileOutputStream.write(bArr, 0, read);
                    progress += read;
                    this.downloadInfo.setProgress(progress);
                    observableEmitter.onNext(this.downloadInfo);
                }
            } catch (Throwable th3) {
                th = th3;
                fileOutputStream2 = fileOutputStream;
                IOUtils.closeQuietly(fileOutputStream2);
                IOUtils.closeQuietly(inputStream);
                throw th;
            }
        }
    }

    private DownloadManager() {
    }

    private DownloadInfo createDownInfo(String str) {
        DownloadInfo downloadInfo = new DownloadInfo(str);
        downloadInfo.setTotal(getContentLength(str));
        downloadInfo.setFileName(str.substring(str.lastIndexOf("/")));
        return downloadInfo;
    }

    private long getContentLength(String str) {
        try {
            Response execute = this.mClient.newCall(new Request.Builder().url(str).build()).execute();
            if (execute != null && execute.isSuccessful()) {
                long contentLength = execute.body().contentLength();
                execute.close();
                if (contentLength == 0) {
                    return -1L;
                }
                return contentLength;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return -1L;
    }

    public static DownloadManager getInstance() {
        AtomicReference<DownloadManager> atomicReference;
        DownloadManager downloadManager;
        do {
            atomicReference = INSTANCE;
            DownloadManager downloadManager2 = atomicReference.get();
            if (downloadManager2 != null) {
                return downloadManager2;
            }
            downloadManager = new DownloadManager();
        } while (!DownloadManager$$ExternalSyntheticBackportWithForwarding0.m(atomicReference, null, downloadManager));
        return downloadManager;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public DownloadInfo getRealFileName(DownloadInfo downloadInfo) {
        String fileName = downloadInfo.getFileName();
        long total = downloadInfo.getTotal();
        File file = new File(MyApplication.getApplication().getFilesDir(), fileName);
        long length = file.exists() ? file.length() : 0L;
        int i = 1;
        while (length >= total) {
            int lastIndexOf = fileName.lastIndexOf(".");
            File file2 = new File(MyApplication.getApplication().getFilesDir(), lastIndexOf == -1 ? fileName + "(" + i + ")" : fileName.substring(0, lastIndexOf) + "(" + i + ")" + fileName.substring(lastIndexOf));
            i++;
            file = file2;
            length = file2.length();
        }
        downloadInfo.setProgress(length);
        downloadInfo.setFileName(file.getName());
        return downloadInfo;
    }

    public void cancel(String str) {
        Call call = this.downCalls.get(str);
        if (call != null) {
            call.cancel();
        }
        this.downCalls.remove(str);
    }

    public void download(String str, DownLoadObserver downLoadObserver) {
        Observable.just(str).filter(new Predicate() { // from class: com.hismith.hismithapp.manager.DownloadManager$$ExternalSyntheticLambda1
            @Override // io.reactivex.functions.Predicate
            public final boolean test(Object obj) {
                return DownloadManager.this.m607lambda$download$0$comhismithhismithappmanagerDownloadManager((String) obj);
            }
        }).flatMap(new Function() { // from class: com.hismith.hismithapp.manager.DownloadManager$$ExternalSyntheticLambda2
            @Override // io.reactivex.functions.Function
            public final Object apply(Object obj) {
                return DownloadManager.this.m608lambda$download$1$comhismithhismithappmanagerDownloadManager((String) obj);
            }
        }).map(new Function() { // from class: com.hismith.hismithapp.manager.DownloadManager$$ExternalSyntheticLambda3
            @Override // io.reactivex.functions.Function
            public final Object apply(Object obj) {
                DownloadInfo realFileName;
                realFileName = DownloadManager.this.getRealFileName((DownloadInfo) obj);
                return realFileName;
            }
        }).flatMap(new Function() { // from class: com.hismith.hismithapp.manager.DownloadManager$$ExternalSyntheticLambda4
            @Override // io.reactivex.functions.Function
            public final Object apply(Object obj) {
                return DownloadManager.this.m609lambda$download$2$comhismithhismithappmanagerDownloadManager((DownloadInfo) obj);
            }
        }).observeOn(AndroidSchedulers.mainThread()).subscribeOn(Schedulers.io()).subscribe(downLoadObserver);
    }

    /* renamed from: lambda$download$0$com-hismith-hismithapp-manager-DownloadManager, reason: not valid java name */
    /* synthetic */ boolean m607lambda$download$0$comhismithhismithappmanagerDownloadManager(String str) throws Exception {
        return !this.downCalls.containsKey(str);
    }

    /* renamed from: lambda$download$1$com-hismith-hismithapp-manager-DownloadManager, reason: not valid java name */
    /* synthetic */ ObservableSource m608lambda$download$1$comhismithhismithappmanagerDownloadManager(String str) throws Exception {
        return Observable.just(createDownInfo(str));
    }

    /* renamed from: lambda$download$2$com-hismith-hismithapp-manager-DownloadManager, reason: not valid java name */
    /* synthetic */ ObservableSource m609lambda$download$2$comhismithhismithappmanagerDownloadManager(DownloadInfo downloadInfo) throws Exception {
        return Observable.create(new DownloadSubscribe(downloadInfo));
    }
}
