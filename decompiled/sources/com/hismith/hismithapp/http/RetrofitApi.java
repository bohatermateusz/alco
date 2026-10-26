package com.hismith.hismithapp.http;

import android.content.Context;
import android.text.TextUtils;
import androidx.media3.exoplayer.ExoPlayer;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.dialog.ProgressDialog;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.model.ChatUpModel;
import com.hismith.hismithapp.model.CheckUpdateModel;
import com.hismith.hismithapp.model.ComplaintReasonsModel;
import com.hismith.hismithapp.model.CountryModel;
import com.hismith.hismithapp.model.FansFocusModel;
import com.hismith.hismithapp.model.FollowModel;
import com.hismith.hismithapp.model.InteractiveModel;
import com.hismith.hismithapp.model.LoginModel;
import com.hismith.hismithapp.model.MessageModel;
import com.hismith.hismithapp.model.MsgSendModel;
import com.hismith.hismithapp.model.PlayModeCreateModel;
import com.hismith.hismithapp.model.PlayModeListModel;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.model.ProductsModel;
import com.hismith.hismithapp.model.ReviewDetailsModel;
import com.hismith.hismithapp.model.ReviewLikeModel;
import com.hismith.hismithapp.model.ReviewSendModel;
import com.hismith.hismithapp.model.TopicListModel;
import com.hismith.hismithapp.model.TopicModel;
import com.hismith.hismithapp.model.TopicReviewModel;
import com.hismith.hismithapp.model.TopicSendModel;
import com.hismith.hismithapp.model.UploadModel;
import com.hismith.hismithapp.model.UserModel;
import com.hismith.hismithapp.model.UserSearchModel;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.CacheUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.disposables.CompositeDisposable;
import io.reactivex.disposables.Disposable;
import io.reactivex.observers.DisposableObserver;
import io.reactivex.schedulers.Schedulers;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.Locale;
import java.util.Map;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;

/* loaded from: classes3.dex */
public class RetrofitApi {
    private static RetrofitApi instance;
    private ApiCallback mApiCallback;
    private Context mContext;
    private ProgressDialog progressBar;
    private boolean isShowProgress = true;
    private RetrofitRequest baseRequest = new RetrofitRequest(RetrofitRequest.EnumRequest.REQUEST_BASE);
    private RetrofitRequest fileRequest = new RetrofitRequest(RetrofitRequest.EnumRequest.REQUEST_FILE);
    public CompositeDisposable mCompositeDisposable = new CompositeDisposable();

    public interface ApiCallback {
        void onComplete();

        void onError(Throwable th);

        void onNext(LoginModel loginModel);

        void onSubscribe(Disposable disposable);
    }

    public RetrofitApi(Context context) {
        this.mContext = context;
        this.progressBar = new ProgressDialog(context);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void createFile(String str) {
        try {
            File file = new File(MyApplication.getApplication().getFilesDir() + "/playSVG");
            if (!file.exists()) {
                file.mkdir();
            }
            File file2 = new File(file, str);
            if (file2.exists()) {
                return;
            }
            file2.createNewFile();
            LogUtil.e(str + "文件创建");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void downloadSVGFile(final String str, final String str2) {
        LogUtil.e("SVG文件下载函数" + str + ", url:" + str2);
        this.baseRequest.getApi().download(str2).subscribeOn(Schedulers.io()).observeOn(Schedulers.newThread()).subscribe(new DisposableObserver<ResponseBody>() { // from class: com.hismith.hismithapp.http.RetrofitApi.51
            @Override // io.reactivex.Observer
            public void onComplete() {
                LogUtil.e("SVG下载完成");
            }

            @Override // io.reactivex.Observer
            public void onError(Throwable th) {
                LogUtil.e("SVG下载响应错误");
            }

            @Override // io.reactivex.Observer
            public void onNext(ResponseBody responseBody) {
                LogUtil.e("SVG下载响应成功" + str2);
                try {
                    InputStream byteStream = responseBody.byteStream();
                    File file = new File(MyApplication.getApplication().getFilesDir() + "/playSVG");
                    if (!file.exists()) {
                        file.mkdir();
                    }
                    File file2 = new File(file, str);
                    if (!file2.exists()) {
                        file2.createNewFile();
                    }
                    FileOutputStream fileOutputStream = new FileOutputStream(file2);
                    byte[] bArr = new byte[2048];
                    while (true) {
                        int read = byteStream.read(bArr);
                        if (read == -1) {
                            fileOutputStream.flush();
                            return;
                        }
                        fileOutputStream.write(bArr, 0, read);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

            @Override // io.reactivex.observers.DisposableObserver
            protected void onStart() {
                super.onStart();
                LogUtil.e(str + " SVG下载请求开始");
                RetrofitApi.this.createFile(str);
            }
        });
    }

    public static RetrofitApi getInstance(Context context) {
        RetrofitApi retrofitApi;
        synchronized (RetrofitApi.class) {
            retrofitApi = instance;
            if (retrofitApi == null) {
                retrofitApi = new RetrofitApi(context);
                instance = retrofitApi;
            }
        }
        return retrofitApi;
    }

    public void allProducts_v1(boolean z) {
        this.baseRequest.getApi().allProducts_v1().subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<ProductsModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.34
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(ProductsModel productsModel) {
                productsModel.getSuccess();
            }
        });
    }

    public void changeMail(String str, String str2, int i, boolean z) {
        this.baseRequest.getApi().changeMail(str, str2, Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.10
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                httpModel.getSuccess();
            }
        });
    }

    public void changePass(String str, String str2, String str3, boolean z) {
        this.baseRequest.getApi().changePass(str, str2, str3).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.14
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                httpModel.getSuccess();
            }
        });
    }

    public void checkUpdate(String str, String str2, int i, boolean z) {
        this.baseRequest.getApi().checkUpdate(str, str2, Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<CheckUpdateModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.19
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(CheckUpdateModel checkUpdateModel) {
                checkUpdateModel.getSuccess();
            }
        });
    }

    public void complaint(String str, int i, String str2, String str3, String str4, boolean z) {
        this.baseRequest.getApi().complaint(str, Integer.valueOf(i), str2, str3, str4).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.23
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                httpModel.getSuccess();
            }
        });
    }

    public void complaintReasons(String str, boolean z) {
        this.baseRequest.getApi().complaintReasons(str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<ComplaintReasonsModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.44
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(ComplaintReasonsModel complaintReasonsModel) {
                complaintReasonsModel.getSuccess();
            }
        });
    }

    public void downloadModeIcon(final ProductModel productModel) {
        new Thread(new Runnable() { // from class: com.hismith.hismithapp.http.RetrofitApi.50
            @Override // java.lang.Runnable
            public void run() {
                if (productModel != null) {
                    try {
                        final File file = new File(MyApplication.getApplication().getFilesDir() + "/modeIcon/" + productModel.getCode());
                        if (file.exists()) {
                            LogUtil.e(file.getName() + "文件夹已存在");
                        } else {
                            if (!file.mkdirs()) {
                                LogUtil.e(file.getName() + "文件夹创建失败");
                                return;
                            }
                            LogUtil.w(file.getName() + "文件夹创建成功");
                        }
                        for (int i = 0; i < productModel.getFuns().length; i++) {
                            int i2 = 0;
                            while (i2 < productModel.getFuns()[i].getModes().length) {
                                Thread.sleep(ExoPlayer.DEFAULT_DETACH_SURFACE_TIMEOUT_MS);
                                final int i3 = i + 1;
                                final int i4 = i2 + 1;
                                final String format = String.format(Locale.getDefault(), "%s第%d个fun，第%d个mode下载", productModel.getCode(), Integer.valueOf(i), Integer.valueOf(i2));
                                AnonymousClass50 anonymousClass50 = this;
                                RetrofitApi.this.baseRequest.getApi().download(productModel.getFuns()[i].getModes()[i2].getmIcon()).subscribeOn(Schedulers.io()).observeOn(Schedulers.newThread()).subscribe(new DisposableObserver<ResponseBody>() { // from class: com.hismith.hismithapp.http.RetrofitApi.50.1
                                    @Override // io.reactivex.Observer
                                    public void onComplete() {
                                        LogUtil.w(format + "完成");
                                    }

                                    @Override // io.reactivex.Observer
                                    public void onError(Throwable th) {
                                        LogUtil.e(format + "错误:" + th.getMessage());
                                    }

                                    @Override // io.reactivex.Observer
                                    public void onNext(ResponseBody responseBody) {
                                        try {
                                            InputStream byteStream = responseBody.byteStream();
                                            File file2 = new File(file, String.format(Locale.getDefault(), "mode%d_%s_%02d.png", Integer.valueOf(i3), productModel.getCode(), Integer.valueOf(i4)));
                                            if (file2.exists()) {
                                                LogUtil.e(file2.getName() + "文件已存在");
                                            } else {
                                                if (!file2.createNewFile()) {
                                                    LogUtil.e(file2.getName() + "文件创建失败");
                                                    return;
                                                }
                                                LogUtil.w(file2.getName() + "文件创建成功");
                                            }
                                            FileOutputStream fileOutputStream = new FileOutputStream(file2);
                                            byte[] bArr = new byte[2048];
                                            while (true) {
                                                int read = byteStream.read(bArr);
                                                if (read == -1) {
                                                    fileOutputStream.flush();
                                                    return;
                                                }
                                                fileOutputStream.write(bArr, 0, read);
                                            }
                                        } catch (Exception e) {
                                            e.printStackTrace();
                                        }
                                    }

                                    @Override // io.reactivex.observers.DisposableObserver
                                    protected void onStart() {
                                        super.onStart();
                                        LogUtil.w(format + "开始");
                                    }
                                });
                                this = anonymousClass50;
                                i2 = i4;
                            }
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        }).start();
    }

    public void emailVerCode(String str, String str2, boolean z) {
        this.baseRequest.getApi().emailVerCode(str, str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.3
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                httpModel.getSuccess();
            }
        });
    }

    public void follow(String str, int i, int i2, boolean z) {
        this.baseRequest.getApi().follow(str, Integer.valueOf(i), Integer.valueOf(i2)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<FollowModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.37
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(FollowModel followModel) {
                followModel.getSuccess();
            }
        });
    }

    public void forgotPass(String str, String str2, int i, boolean z) {
        this.baseRequest.getApi().forgotPass(str, str2, Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<LoginModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.5
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(LoginModel loginModel) {
                loginModel.getSuccess();
            }
        });
    }

    public void getCountry(boolean z) {
        this.baseRequest.getApi().getCountry().subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<CountryModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.11
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(CountryModel countryModel) {
                countryModel.getSuccess();
            }
        });
    }

    public void getFansFocus(String str, String str2, int i, boolean z) {
        this.baseRequest.getApi().getFansFocus(str, str2, Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<FansFocusModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.38
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(FansFocusModel fansFocusModel) {
                fansFocusModel.getSuccess();
            }
        });
    }

    public void getMyInfo(String str, boolean z) {
        this.baseRequest.getApi().getMyInfo(str, MyApplication.FCM_TOKEN, AppUtil.getAppVersionName(MyApplication.getApplication())).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<UserModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.17
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(UserModel userModel) {
                userModel.getSuccess();
            }
        });
    }

    public void getProduct(String str, String str2, boolean z) {
        this.baseRequest.getApi().getProduct(str, str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<ProductModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.33
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(ProductModel productModel) {
                productModel.getSuccess();
            }
        });
    }

    public void getUserInfo(String str, int i, boolean z) {
        this.baseRequest.getApi().getUserInfo(str, Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<UserModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.18
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(UserModel userModel) {
                userModel.getSuccess();
            }
        });
    }

    public void login(String str, String str2, boolean z) {
        this.baseRequest.getApi().login(str, str2, "app").subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<LoginModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.1
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(LoginModel loginModel) {
                loginModel.getSuccess();
            }
        });
    }

    public void messageNew(String str, int i, boolean z) {
        this.baseRequest.getApi().messageNew(str, Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<MessageModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.41
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(MessageModel messageModel) {
                messageModel.getSuccess();
            }
        });
    }

    public void msgFirstChat(String str, int i, boolean z) {
        this.baseRequest.getApi().msgFirstChat(str, Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<ChatUpModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.40
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(ChatUpModel chatUpModel) {
                chatUpModel.getSuccess();
            }
        });
    }

    public void msgInteractive(String str, int i, boolean z) {
        this.baseRequest.getApi().msgInteractive(str, Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<InteractiveModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.36
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(InteractiveModel interactiveModel) {
                interactiveModel.getSuccess();
            }
        });
    }

    public void msgSend(String str, String str2, Integer num, String str3, Integer num2, String str4, int i, boolean z) {
        this.baseRequest.getApi().msgSend(str, str2, num, str3, num2, str4, Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<MsgSendModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.42
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(MsgSendModel msgSendModel) {
                msgSendModel.getSuccess();
            }
        });
    }

    public void playModeAction(String str, int i, String str2, String str3, boolean z) {
        this.baseRequest.getApi().playModeAction(str, i, str3, str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.47
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                httpModel.getSuccess();
            }
        });
    }

    public void playModeAction(String str, int i, String str2, boolean z) {
        this.baseRequest.getApi().playModeAction(str, i, str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.46
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                httpModel.getSuccess();
            }
        });
    }

    public void playModeList(String str, String str2, int i, String str3, boolean z) {
        this.baseRequest.getApi().playModeList(str, str2, MyApplication.getApplication().getDeviceCode(), i, str3).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<PlayModeListModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.45
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(PlayModeListModel playModeListModel) {
                playModeListModel.getSuccess();
            }
        });
    }

    public void playModeSave(String str, String str2, String str3, String str4, String str5, String str6, boolean z) {
        this.baseRequest.getApi().playModeSave(str, str2, str3, str4, str5, str6).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<PlayModeCreateModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.48
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(PlayModeCreateModel playModeCreateModel) {
                playModeCreateModel.getSuccess();
            }
        });
    }

    public void presenterSVGFile(ProductModel productModel) {
        String str;
        LogUtil.e("================SVG逻辑判断函数==================");
        if (productModel == null || TextUtils.isEmpty(productModel.getSvgName()) || TextUtils.isEmpty(productModel.getSvgUrl())) {
            LogUtil.e("SVG信息数据中有空。无法下载 code : " + (productModel != null ? productModel.getCode() : ""));
            return;
        }
        String svgName = productModel.getSvgName();
        if (productModel.getSvgLeftRight() == 1) {
            svgName = svgName + "_left";
            str = productModel.getSvgName() + "_right";
        }
        long playSVGModifyTime = CacheUtil.getPlaySVGModifyTime(MyApplication.getApplication(), svgName);
        if (CacheUtil.playSVGExists(MyApplication.getApplication(), svgName) && ((productModel.getSvgLeftRight() != 1 || CacheUtil.playSVGExists(MyApplication.getApplication(), str)) && (productModel.getSvgVersion() <= 0 || productModel.getSvgVersion() * 1000 <= playSVGModifyTime))) {
            LogUtil.e(svgName + " SVG文件已存在，并且已经是最新版。无需下载");
            return;
        }
        if (CacheUtil.playSVGExists(MyApplication.getApplication(), svgName)) {
            LogUtil.e(svgName + " SVG本地文件:" + playSVGModifyTime + ", server最后修改时间：" + (productModel.getSvgVersion() * 1000));
            LogUtil.e("SVG文件时间判断不是最新版，需要重新下载");
        } else {
            LogUtil.e(svgName + " SVG文件不存在，需要下载");
        }
        downloadSVGFile(svgName + ".svg", productModel.getSvgUrl());
        if (productModel.getSvgLeftRight() == 1) {
            downloadSVGFile(str + ".svg", productModel.getSvgUrl().replace("_left", "_right"));
        }
    }

    public void register(String str, String str2, int i, boolean z) {
        this.baseRequest.getApi().register(str, str2, Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<LoginModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.4
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(LoginModel loginModel) {
                loginModel.getSuccess();
            }
        });
    }

    public void setChatUpState(String str, int i, boolean z) {
        this.baseRequest.getApi().setChatUpState(str, Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.16
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                httpModel.getSuccess();
            }
        });
    }

    public void setRemind(String str, int i, int i2, int i3, boolean z) {
        this.baseRequest.getApi().setRemind(str, Integer.valueOf(i), Integer.valueOf(i2), Integer.valueOf(i3)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.15
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                httpModel.getSuccess();
            }
        });
    }

    public void setUserBlackList(String str, int i, boolean z) {
        this.baseRequest.getApi().setUserBlackList(str, Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.43
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                httpModel.getSuccess();
            }
        });
    }

    public void setUserInfo(String str, int i, String str2, int i2, int i3, String str3, String str4, String str5, String str6, int i4, String str7, int i5, int i6, int i7, int i8, int i9, boolean z) {
        this.baseRequest.getApi().setUserInfo(str, Integer.valueOf(i), str2, Integer.valueOf(i2), Integer.valueOf(i3), str3, str4, str5, str6, Integer.valueOf(i4), str7, Integer.valueOf(i5), Integer.valueOf(i6), Integer.valueOf(i7), Integer.valueOf(i8), Integer.valueOf(i9)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.6
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                httpModel.getSuccess();
            }
        });
    }

    public void setUserInfo(String str, String str2, int i, int i2, int i3, String str3, String str4, String str5, boolean z) {
        this.baseRequest.getApi().setUserInfo(str, str2, Integer.valueOf(i), Integer.valueOf(i2), Integer.valueOf(i3), str3, str4, str5).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.7
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                httpModel.getSuccess();
            }
        });
    }

    public void setUserInfo(String str, String str2, int i, int i2, String str3, int i3, int i4, String str4, boolean z) {
        this.baseRequest.getApi().setUserInfo(str, str2, Integer.valueOf(i), Integer.valueOf(i2), str3, Integer.valueOf(i3), Integer.valueOf(i4), str4).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.8
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                httpModel.getSuccess();
            }
        });
    }

    public void setUserInfo(String str, String str2, boolean z) {
        this.baseRequest.getApi().setUserInfo(str, str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.9
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                httpModel.getSuccess();
            }
        });
    }

    public void setUserLang(String str, String str2, boolean z) {
        this.baseRequest.getApi().setUserLang(str, str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.13
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                httpModel.getSuccess();
            }
        });
    }

    public void setVoice(String str, String str2, int i, boolean z) {
        this.baseRequest.getApi().setVoice(str, str2, Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.35
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                httpModel.getSuccess();
            }
        });
    }

    public void submitError(String str) {
        this.baseRequest.getApi().submitError((String) SharedPreferencesUtil.get(MyApplication.getApplication(), Constans.Token), "android", AppUtil.getAppVersionName(MyApplication.getApplication()), str).subscribeOn(Schedulers.io()).observeOn(Schedulers.io()).subscribe(new RxSubscribe<HttpModel>(this.mContext, null, false) { // from class: com.hismith.hismithapp.http.RetrofitApi.49
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                httpModel.getSuccess();
            }
        });
    }

    public void topicAction(String str, int i, String str2, boolean z) {
        this.baseRequest.getApi().topicAction(str, Integer.valueOf(i), str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.25
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                httpModel.getSuccess();
            }
        });
    }

    public void topicDel(String str, int i, boolean z) {
        this.baseRequest.getApi().topicDel(str, Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.22
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                httpModel.getSuccess();
            }
        });
    }

    public void topicDetails(String str, int i, boolean z) {
        this.baseRequest.getApi().topicDetails(str, Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.26
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicModel topicModel) {
                topicModel.getSuccess();
            }
        });
    }

    public void topicEdit(String str, int i, String str2, String str3, String str4, boolean z) {
        this.baseRequest.getApi().topicEdit(str, Integer.valueOf(i), str2, str3, str4).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicSendModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.21
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicSendModel topicSendModel) {
                topicSendModel.getSuccess();
            }
        });
    }

    public void topicList(String str, String str2, int i, String str3, boolean z) {
        this.baseRequest.getApi().topicList(str, str2, Integer.valueOf(i), str3).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicListModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.24
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicListModel topicListModel) {
                topicListModel.getSuccess();
            }
        });
    }

    public void topicReviewDel(String str, int i, boolean z) {
        this.baseRequest.getApi().topicReviewDel(str, Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.32
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                httpModel.getSuccess();
            }
        });
    }

    public void topicReviewDetails(String str, int i, int i2, boolean z) {
        this.baseRequest.getApi().topicReviewDetails(str, Integer.valueOf(i), Integer.valueOf(i2)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<ReviewDetailsModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.30
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(ReviewDetailsModel reviewDetailsModel) {
                reviewDetailsModel.getSuccess();
            }
        });
    }

    public void topicReviewLike(String str, int i, String str2, boolean z) {
        this.baseRequest.getApi().topicReviewLike(str, Integer.valueOf(i), str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<ReviewLikeModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.31
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(ReviewLikeModel reviewLikeModel) {
                reviewLikeModel.getSuccess();
            }
        });
    }

    public void topicReviewReplySend(String str, int i, String str2, int i2, boolean z) {
        this.baseRequest.getApi().topicReviewReplySend(str, Integer.valueOf(i), str2, Integer.valueOf(i2)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<ReviewSendModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.28
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(ReviewSendModel reviewSendModel) {
                reviewSendModel.getSuccess();
            }
        });
    }

    public void topicReviewSend(String str, int i, String str2, boolean z) {
        this.baseRequest.getApi().topicReviewSend(str, Integer.valueOf(i), str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<ReviewSendModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.27
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(ReviewSendModel reviewSendModel) {
                reviewSendModel.getSuccess();
            }
        });
    }

    public void topicReviews(String str, int i, int i2, boolean z) {
        this.baseRequest.getApi().topicReviews(str, Integer.valueOf(i), Integer.valueOf(i2)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicReviewModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.29
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicReviewModel topicReviewModel) {
                topicReviewModel.getSuccess();
            }
        });
    }

    public void topicSend(String str, String str2, String str3, String str4, String str5, boolean z) {
        this.baseRequest.getApi().topicSend(str, str2, str3, str4, str5).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicSendModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.20
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicSendModel topicSendModel) {
                topicSendModel.getSuccess();
            }
        });
    }

    public void uploadFile(Map<String, RequestBody> map, boolean z) {
        this.baseRequest.getApi().uploadFile(map).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<UploadModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.12
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(UploadModel uploadModel) {
                uploadModel.getSuccess();
            }
        });
    }

    public void userOauthLogin(String str, String str2, boolean z) {
        this.baseRequest.getApi().userOauthLogin(str, str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<LoginModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.2
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(LoginModel loginModel) {
                loginModel.getSuccess();
            }
        });
    }

    public void userSearch(String str, String str2, int i, boolean z) {
        this.baseRequest.getApi().userSearch(str, str2, Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<UserSearchModel>(this.mContext, this.progressBar, z) { // from class: com.hismith.hismithapp.http.RetrofitApi.39
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(UserSearchModel userSearchModel) {
                userSearchModel.getSuccess();
            }
        });
    }
}
