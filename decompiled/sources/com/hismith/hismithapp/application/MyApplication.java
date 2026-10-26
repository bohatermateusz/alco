package com.hismith.hismithapp.application;

import android.app.Application;
import android.content.Context;
import android.content.Intent;
import android.content.res.Configuration;
import android.text.TextUtils;
import androidx.lifecycle.ProcessLifecycleOwner;
import com.facebook.FacebookSdk;
import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.FirebaseApp;
import com.google.firebase.messaging.FirebaseMessaging;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitApi;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.manager.PiuTimerManager;
import com.hismith.hismithapp.manager.UpgradeManager;
import com.hismith.hismithapp.model.CheckUpdateModel;
import com.hismith.hismithapp.model.ComplaintReasonsModel;
import com.hismith.hismithapp.model.LanguageModel;
import com.hismith.hismithapp.model.PlayLogsModel;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.model.ProductsModel;
import com.hismith.hismithapp.model.UserModel;
import com.hismith.hismithapp.observer.AppBackgroundObserver;
import com.hismith.hismithapp.picker.PictureSelectorManager;
import com.hismith.hismithapp.utils.AppManager;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.CacheUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.DialogUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.scwang.smart.refresh.footer.BallPulseFooter;
import com.scwang.smart.refresh.header.BezierRadarHeader;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshFooter;
import com.scwang.smart.refresh.layout.api.RefreshHeader;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.DefaultRefreshFooterCreator;
import com.scwang.smart.refresh.layout.listener.DefaultRefreshHeaderCreator;
import com.tencent.mm.opensdk.openapi.WXAPIFactory;
import com.twitter.sdk.android.core.DefaultLogger;
import com.twitter.sdk.android.core.Twitter;
import com.twitter.sdk.android.core.TwitterAuthConfig;
import com.twitter.sdk.android.core.TwitterConfig;
import io.reactivex.Observer;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.disposables.Disposable;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.Iterator;
import okhttp3.ResponseBody;
import org.json.JSONException;
import org.json.JSONObject;

/* loaded from: classes3.dex */
public class MyApplication extends Application {
    public static String FCM_TOKEN = "";
    private static MyApplication myApplication;
    private RetrofitRequest baseRequest;
    private JSONObject languageJson;
    private UserModel mHismithModel;
    private ProductModel mProductModel;
    private UserModel mUserModel;
    private String requestId;
    private Integer userId;
    private String deviceCode = "";
    private String deviceName = "";
    private String deviceAddress = "";
    private boolean isNewCheckUpdate = false;
    private boolean isBluetoothShowGuide = true;
    private boolean isLineCreateShowGuide = true;

    private void allProducts() {
        this.baseRequest.getApi().allProducts_v1().subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<ProductsModel>(this, null, false) { // from class: com.hismith.hismithapp.application.MyApplication.6
            /* JADX INFO: Access modifiers changed from: protected */
            /* JADX WARN: Code restructure failed: missing block: B:17:0x0066, code lost:
            
                com.hismith.hismithapp.application.MyApplication.getApplication().setmProductModel(r3.getProducts()[r2]);
             */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            /*
                Code decompiled incorrectly, please refer to instructions dump.
            */
            public void onSuccess(ProductsModel productsModel) {
                if (productsModel.getSuccess() == 1) {
                    SharedPreferencesUtil.put(MyApplication.myApplication, Constans.ProductAllData, new Gson().toJson(productsModel));
                    try {
                        if (productsModel.getProducts() != null && productsModel.getProducts().length > 0) {
                            int i = 0;
                            while (true) {
                                if (i >= productsModel.getProducts().length) {
                                    break;
                                }
                                RetrofitApi.getInstance(MyApplication.getApplication()).presenterSVGFile(productsModel.getProducts()[i]);
                                if (BleManager.getInstance(MyApplication.myApplication).getStatus() == 2 && productsModel.getProducts()[i].getCode().equals(MyApplication.getApplication().getDeviceCode())) {
                                    break;
                                } else {
                                    i++;
                                }
                            }
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    MyApplication.myApplication.sendBroadcast(new Intent(Constans.RECEIVER_PRODUCT_INFO));
                }
            }
        });
    }

    private void checkUpdate() {
        SharedPreferencesUtil.remove(myApplication, Constans.newVersion);
        SharedPreferencesUtil.remove(myApplication, Constans.newVersionForcibly);
        SharedPreferencesUtil.remove(myApplication, Constans.newVersionPrompt);
        SharedPreferencesUtil.remove(myApplication, Constans.newVersionDesc);
        final String readJson = CacheUtil.readJson(myApplication, Constans.LanguageFileName);
        String versionAllLangs = LanguageUtil.getVersionAllLangs(readJson);
        this.baseRequest.getApi().checkUpdate((String) SharedPreferencesUtil.get(getApplication(), Constans.Token), versionAllLangs, 0, "Android", AppUtil.getAppVersionName(myApplication)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<CheckUpdateModel>(myApplication, null, false) { // from class: com.hismith.hismithapp.application.MyApplication.5
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                if (MyApplication.this.isNewCheckUpdate) {
                    MyApplication.this.isNewCheckUpdate = false;
                    MyApplication.this.initLanguageList();
                    MyApplication.this.initLanguage();
                }
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(CheckUpdateModel checkUpdateModel) {
                if (checkUpdateModel.getSuccess() == 1) {
                    try {
                        SharedPreferencesUtil.put(MyApplication.myApplication, Constans.storeUrl, checkUpdateModel.getStoreUrl());
                        SharedPreferencesUtil.put(MyApplication.myApplication, Constans.contactUserId, Integer.valueOf(checkUpdateModel.getContactUserId()));
                        if (checkUpdateModel.getRoomVoice() != null) {
                            SharedPreferencesUtil.put(MyApplication.myApplication, Constans.roomVoice, checkUpdateModel.getRoomVoice());
                        }
                        if (checkUpdateModel.getToken() != null) {
                            SharedPreferencesUtil.put(MyApplication.myApplication, Constans.tempToken, checkUpdateModel.getToken());
                        }
                        if (checkUpdateModel.getAboutApp() != null) {
                            SharedPreferencesUtil.put(MyApplication.myApplication, Constans.aboutApp, checkUpdateModel.getAboutApp());
                        }
                        if (checkUpdateModel.getCopyright() != null) {
                            SharedPreferencesUtil.put(MyApplication.myApplication, Constans.copyright, checkUpdateModel.getCopyright());
                        }
                        if (checkUpdateModel.getMainBackgroundUrl() != null) {
                            SharedPreferencesUtil.put(MyApplication.myApplication, Constans.mainBackgroundUrl, checkUpdateModel.getMainBackgroundUrl());
                        }
                        if (checkUpdateModel.getNewVersion() != null) {
                            SharedPreferencesUtil.put(MyApplication.myApplication, Constans.newVersion, checkUpdateModel.getNewVersion());
                            SharedPreferencesUtil.put(MyApplication.myApplication, Constans.newVersionForcibly, Integer.valueOf(checkUpdateModel.getNewVersionForcibly()));
                            SharedPreferencesUtil.put(MyApplication.myApplication, Constans.newVersionPrompt, Integer.valueOf(checkUpdateModel.getNewVersionPrompt()));
                            SharedPreferencesUtil.put(MyApplication.myApplication, Constans.newVersionDesc, checkUpdateModel.getNewVersionDesc());
                            if (AppManager.getInstance().currentActivity() != null && !"LunchActivity".equals(AppManager.getInstance().currentActivity().getLocalClassName())) {
                                UpgradeManager.initUpgrade(AppManager.getInstance().currentActivity());
                            }
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    if (checkUpdateModel.getAllLangs() == null || !checkUpdateModel.getAllLangs().isJsonObject()) {
                        return;
                    }
                    try {
                        JsonArray jsonArray = (JsonArray) new Gson().fromJson(readJson, JsonArray.class);
                        if (jsonArray == null) {
                            jsonArray = new JsonArray();
                        }
                        JsonObject asJsonObject = checkUpdateModel.getAllLangs().getAsJsonObject();
                        for (String str : asJsonObject.keySet()) {
                            if (asJsonObject.get(str) == null || TextUtils.isEmpty(asJsonObject.get(str).toString()) || "null".equals(asJsonObject.get(str).toString())) {
                                CacheUtil.deleteJson(MyApplication.myApplication, str);
                            } else {
                                JsonObject asJsonObject2 = asJsonObject.get(str).getAsJsonObject();
                                CacheUtil.writeJson(MyApplication.myApplication, asJsonObject2.toString(), str, false);
                                asJsonObject2.remove("langs");
                                boolean z = true;
                                for (int i = 0; i < jsonArray.size(); i++) {
                                    if (jsonArray.get(i) != null && jsonArray.get(i).isJsonObject()) {
                                        JsonObject asJsonObject3 = jsonArray.get(i).getAsJsonObject();
                                        if (str.equals(asJsonObject3.get("iso") == null ? "" : asJsonObject3.get("iso").getAsString()) && asJsonObject.get(str).isJsonObject()) {
                                            jsonArray.set(i, asJsonObject2);
                                            z = false;
                                        }
                                    }
                                }
                                if (z) {
                                    jsonArray.add(asJsonObject2);
                                }
                            }
                            MyApplication.this.isNewCheckUpdate = true;
                        }
                        CacheUtil.writeJson(MyApplication.myApplication, jsonArray.toString(), Constans.LanguageFileName, false);
                    } catch (Exception e2) {
                        e2.printStackTrace();
                    }
                }
            }
        });
    }

    public static MyApplication getApplication() {
        return myApplication;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void getUserInfo() {
        String str = (String) SharedPreferencesUtil.get(getApplication(), Constans.Token);
        if (str == null || "".equals(str)) {
            return;
        }
        this.baseRequest.getApi().getMyInfo(str, FCM_TOKEN, AppUtil.getAppVersionName(getApplication())).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<UserModel>(getApplication(), null, false) { // from class: com.hismith.hismithapp.application.MyApplication.4
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(UserModel userModel) {
                if (userModel.getSuccess() == 1) {
                    MyApplication.getApplication().setmUserModel(userModel);
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initLanguageList() {
        LogUtil.e("initLanguageList()");
        ArrayList arrayList = new ArrayList();
        String readJson = CacheUtil.readJson(myApplication, Constans.LanguageFileName);
        if (TextUtils.isEmpty(readJson)) {
            return;
        }
        try {
            Gson gson = new Gson();
            JsonElement jsonElement = (JsonElement) gson.fromJson(readJson, JsonElement.class);
            if (jsonElement.isJsonObject()) {
                JsonArray jsonArray = new JsonArray();
                Iterator<String> it = jsonElement.getAsJsonObject().keySet().iterator();
                while (it.hasNext()) {
                    String readJson2 = CacheUtil.readJson(myApplication, it.next());
                    if (!TextUtils.isEmpty(readJson2)) {
                        JsonElement jsonElement2 = (JsonElement) gson.fromJson(readJson2, JsonElement.class);
                        if (jsonElement2.isJsonObject()) {
                            JsonObject asJsonObject = jsonElement2.getAsJsonObject();
                            asJsonObject.remove("langs");
                            LogUtil.e("langJsonObject:" + asJsonObject);
                            arrayList.add(new LanguageModel(asJsonObject.get("iso") == null ? "" : asJsonObject.get("iso").getAsString(), asJsonObject.get("version") == null ? "" : asJsonObject.get("version").getAsString(), asJsonObject.get("name") == null ? "" : asJsonObject.get("name").getAsString()));
                            jsonArray.add(asJsonObject);
                        }
                    }
                }
                CacheUtil.writeJson(myApplication, jsonArray.toString(), Constans.LanguageFileName, false);
            } else if (jsonElement.isJsonArray()) {
                JsonArray asJsonArray = jsonElement.getAsJsonArray();
                for (int i = 0; i < asJsonArray.size(); i++) {
                    if (asJsonArray.get(i).isJsonObject()) {
                        JsonObject asJsonObject2 = asJsonArray.get(i).getAsJsonObject();
                        arrayList.add(new LanguageModel(asJsonObject2.get("iso") == null ? "" : asJsonObject2.get("iso").getAsString(), asJsonObject2.get("version") == null ? "" : asJsonObject2.get("version").getAsString(), asJsonObject2.get("name") == null ? "" : asJsonObject2.get("name").getAsString()));
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        LanguageUtil.setLanguageList(arrayList);
    }

    private void initSmartRefreshLayout() {
        SmartRefreshLayout.setDefaultRefreshHeaderCreator(new DefaultRefreshHeaderCreator() { // from class: com.hismith.hismithapp.application.MyApplication.2
            @Override // com.scwang.smart.refresh.layout.listener.DefaultRefreshHeaderCreator
            public RefreshHeader createRefreshHeader(Context context, RefreshLayout refreshLayout) {
                refreshLayout.setPrimaryColorsId(R.color.transparent, R.color.smartRefreshPrimaryColor);
                return new BezierRadarHeader(context);
            }
        });
        SmartRefreshLayout.setDefaultRefreshFooterCreator(new DefaultRefreshFooterCreator() { // from class: com.hismith.hismithapp.application.MyApplication.3
            @Override // com.scwang.smart.refresh.layout.listener.DefaultRefreshFooterCreator
            public RefreshFooter createRefreshFooter(Context context, RefreshLayout refreshLayout) {
                BallPulseFooter ballPulseFooter = new BallPulseFooter(context);
                ballPulseFooter.setAnimatingColor(MyApplication.this.getResources().getColor(R.color.app_style_color));
                return ballPulseFooter;
            }
        });
    }

    private void initTwitter() {
        Twitter.initialize(new TwitterConfig.Builder(this).logger(new DefaultLogger(3)).twitterAuthConfig(new TwitterAuthConfig(getString(R.string.twitter_consumer_key), getString(R.string.twitter_consumer_secret))).debug(true).build());
    }

    private void languageWork() {
        LanguageUtil.getCurrLanguage(this);
    }

    private void resetServer() {
        this.baseRequest.getApi().resetServer("https://www.hismiths.com/server").subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new Observer<ResponseBody>() { // from class: com.hismith.hismithapp.application.MyApplication.7
            @Override // io.reactivex.Observer
            public void onComplete() {
            }

            @Override // io.reactivex.Observer
            public void onError(Throwable th) {
            }

            @Override // io.reactivex.Observer
            public void onNext(ResponseBody responseBody) {
            }

            @Override // io.reactivex.Observer
            public void onSubscribe(Disposable disposable) {
            }
        });
    }

    @Override // android.content.ContextWrapper
    protected void attachBaseContext(Context context) {
        super.attachBaseContext(context);
    }

    public void complaintReasons(String str) {
        this.baseRequest.getApi().complaintReasons(str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<ComplaintReasonsModel>(myApplication, null, false) { // from class: com.hismith.hismithapp.application.MyApplication.8
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(ComplaintReasonsModel complaintReasonsModel) {
                if (complaintReasonsModel.getSuccess() == 1 && complaintReasonsModel.getComplaintReasons() != null && complaintReasonsModel.getComplaintReasons().isJsonObject()) {
                    SharedPreferencesUtil.put(MyApplication.myApplication, Constans.complaintReasons, complaintReasonsModel.getComplaintReasons().getAsJsonObject().toString());
                    for (String str2 : complaintReasonsModel.getComplaintReasons().getAsJsonObject().keySet()) {
                        LogUtil.e("举报理由：key:" + str2 + ", value:" + complaintReasonsModel.getComplaintReasons().getAsJsonObject().get(str2).getAsString());
                    }
                }
            }
        });
    }

    public String getDeviceAddress() {
        return this.deviceAddress;
    }

    public String getDeviceCode() {
        return this.deviceCode;
    }

    public String getDeviceName() {
        return this.deviceName;
    }

    public JSONObject getLanguageJson() {
        return this.languageJson;
    }

    public String getRequestId() {
        return this.requestId;
    }

    public Integer getUserId() {
        return this.userId;
    }

    public UserModel getmHismithModel() {
        return this.mHismithModel;
    }

    public ProductModel getmProductModel() {
        return this.mProductModel;
    }

    public UserModel getmUserModel() {
        return this.mUserModel;
    }

    public void initLanguage() {
        LogUtil.e("initLanguage()");
        try {
            String readJson = CacheUtil.readJson(myApplication, LanguageUtil.getCurrLanguage(myApplication));
            if (!TextUtils.isEmpty(readJson)) {
                this.languageJson = new JSONObject(readJson);
            }
        } catch (JSONException e) {
            e.printStackTrace();
        } catch (Exception e2) {
            e2.printStackTrace();
        }
        try {
            PictureSelectorManager.refreshLanguage(myApplication);
        } catch (Exception e3) {
            e3.printStackTrace();
        }
    }

    public void initLanguage(String str) {
        LogUtil.e("initLanguage()");
        try {
            String readJson = CacheUtil.readJson(myApplication, str);
            if (TextUtils.isEmpty(readJson)) {
                return;
            }
            this.languageJson = new JSONObject(readJson);
        } catch (JSONException e) {
            e.printStackTrace();
        } catch (Exception e2) {
            e2.printStackTrace();
        }
    }

    public boolean isBluetoothShowGuide() {
        return this.isBluetoothShowGuide;
    }

    public boolean isLineCreateShowGuide() {
        return this.isLineCreateShowGuide;
    }

    @Override // android.app.Application, android.content.ComponentCallbacks
    public void onConfigurationChanged(Configuration configuration) {
        super.onConfigurationChanged(configuration);
        languageWork();
    }

    @Override // android.app.Application
    public void onCreate() {
        super.onCreate();
        myApplication = this;
        ProcessLifecycleOwner.get().getLifecycle().addObserver(new AppBackgroundObserver());
        initSmartRefreshLayout();
        registerActivityLifecycleCallbacks(new MyLifecycleHandler());
        FirebaseApp.initializeApp(myApplication);
        FirebaseMessaging.getInstance().getToken().addOnCompleteListener(new OnCompleteListener<String>() { // from class: com.hismith.hismithapp.application.MyApplication.1
            @Override // com.google.android.gms.tasks.OnCompleteListener
            public void onComplete(Task<String> task) {
                if (task.isSuccessful()) {
                    String result = task.getResult();
                    MyApplication.FCM_TOKEN = result;
                    MyApplication.this.getUserInfo();
                    LogUtil.e("fcm getInstanceId():" + result);
                }
            }
        });
        FirebaseMessaging.getInstance().setAutoInitEnabled(true);
        FacebookSdk.sdkInitialize(getApplicationContext());
        initTwitter();
        WXAPIFactory.createWXAPI(this, getString(R.string.wx_id), true).registerApp(getString(R.string.wx_id));
        this.baseRequest = new RetrofitRequest(RetrofitRequest.EnumRequest.REQUEST_BASE);
        PiuTimerManager.getInstance(myApplication);
        languageWork();
        initLanguageList();
        initLanguage();
        checkUpdate();
        allProducts();
    }

    @Override // android.app.Application, android.content.ComponentCallbacks
    public void onLowMemory() {
        super.onLowMemory();
        LogUtil.e("Application onLowMemory()");
        AppManager.getInstance().finishAllActivity();
    }

    @Override // android.app.Application
    public void onTerminate() {
        super.onTerminate();
        LogUtil.e("Application onTerminate()");
    }

    public void setBluetoothShowGuide(boolean z) {
        this.isBluetoothShowGuide = z;
    }

    public void setDeviceAddress(String str) {
        this.deviceAddress = str;
    }

    public void setDeviceCode(String str) {
        this.deviceCode = str;
    }

    public void setDeviceName(String str) {
        this.deviceName = str;
    }

    public void setLineCreateShowGuide(boolean z) {
        this.isLineCreateShowGuide = z;
    }

    public void setPlayLogs(String str, int i) {
        String str2 = (String) SharedPreferencesUtil.get(getApplication(), Constans.Token);
        if (TextUtils.isEmpty(str2) || TextUtils.isEmpty(str)) {
            return;
        }
        this.baseRequest.getApi().setPlayLogs(str2, str, i).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<PlayLogsModel>(myApplication, null, false) { // from class: com.hismith.hismithapp.application.MyApplication.11
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(PlayLogsModel playLogsModel) {
                if (playLogsModel.getSuccess() != 1 || TextUtils.isEmpty(playLogsModel.getOverMsg())) {
                    return;
                }
                DialogUtil.showTipsDialog(AppManager.getInstance().currentActivity(), null, playLogsModel.getOverMsg(), null);
            }
        });
    }

    public void setRequestId(String str) {
        this.requestId = str;
    }

    public void setUserId(Integer num) {
        this.userId = num;
    }

    public void setmHismithModel(UserModel userModel) {
        this.mHismithModel = userModel;
    }

    public void setmProductModel(ProductModel productModel) {
        this.mProductModel = productModel;
    }

    public void setmUserModel(UserModel userModel) {
        this.mUserModel = userModel;
    }

    public void submitError(String str) {
        this.baseRequest.getApi().submitError((String) SharedPreferencesUtil.get(getApplication(), Constans.Token), "android", AppUtil.getAppVersionName(myApplication), str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(myApplication, null, false) { // from class: com.hismith.hismithapp.application.MyApplication.10
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
            }
        });
    }

    public void submitError(String str, String str2) {
        this.baseRequest.getApi().submitError(str, "android", AppUtil.getAppVersionName(myApplication), str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(myApplication, null, false) { // from class: com.hismith.hismithapp.application.MyApplication.9
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
            }
        });
    }
}
