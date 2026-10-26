package com.hismith.hismithapp;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.view.View;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.TextView;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.CacheUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.view.CustomWebView;
import java.io.File;

/* loaded from: classes3.dex */
public class AdsActivity extends BaseActivity implements View.OnClickListener {
    private TextView txt_skip;
    private CustomWebView web_ads;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.AdsActivity.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            if (i == 0) {
                AdsActivity.this.startActivity(new Intent(AdsActivity.this, (Class<?>) LoginActivity.class));
                AdsActivity.this.finish();
                return false;
            }
            if (i == 1) {
                Intent intent = new Intent(AdsActivity.this, (Class<?>) MainActivity.class);
                if (AdsActivity.this.getIntent() != null && AdsActivity.this.getIntent().getExtras() != null) {
                    intent.putExtras(AdsActivity.this.getIntent().getExtras());
                }
                AdsActivity.this.startActivity(intent);
                AdsActivity.this.finish();
                return false;
            }
            if (i != 2 || AdsActivity.this.skipTime <= 0) {
                return false;
            }
            AdsActivity.this.txt_skip.setText(AdsActivity.this.skipTime + " " + LanguageUtil.getLanguageByKey(AdsActivity.this, "skip"));
            AdsActivity.access$010(AdsActivity.this);
            AdsActivity.this.mHandler.sendEmptyMessageDelayed(2, 1000L);
            return false;
        }
    });
    private String showAdsVersion = "";
    private int skipTime = 5;
    private int animTime = 5 * 1000;

    class GameWebViewClient extends WebViewClient {
        GameWebViewClient() {
        }

        @Override // android.webkit.WebViewClient
        public void onPageFinished(WebView webView, String str) {
            super.onPageFinished(webView, str);
        }

        @Override // android.webkit.WebViewClient
        public void onPageStarted(WebView webView, String str, Bitmap bitmap) {
            super.onPageStarted(webView, str, bitmap);
        }

        @Override // android.webkit.WebViewClient
        public boolean shouldOverrideUrlLoading(WebView webView, String str) {
            LogUtil.e("shouldOverrideUrlLoading url:" + str);
            try {
                AppUtil.startUriToWeb(AdsActivity.this, str);
            } catch (Exception unused) {
            }
            return true;
        }
    }

    static /* synthetic */ int access$010(AdsActivity adsActivity) {
        int i = adsActivity.skipTime;
        adsActivity.skipTime = i - 1;
        return i;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() == R.id.txt_skip) {
            boolean isEmpty = TextUtils.isEmpty((String) SharedPreferencesUtil.get(this, Constans.Token, ""));
            Handler handler = this.mHandler;
            if (isEmpty) {
                handler.removeMessages(0);
                this.mHandler.sendEmptyMessage(0);
            } else {
                handler.removeMessages(1);
                this.mHandler.sendEmptyMessage(1);
            }
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        File[] listFiles;
        super.onCreate(bundle);
        setContentView(R.layout.activity_ads);
        TextView textView = (TextView) findViewById(R.id.txt_skip);
        this.txt_skip = textView;
        textView.setOnClickListener(this);
        this.web_ads = (CustomWebView) findViewById(R.id.web_ads);
        this.web_ads.getSettings().setCacheMode(1);
        this.web_ads.setBackgroundColor(0);
        this.web_ads.setWebViewClient(new GameWebViewClient());
        this.web_ads.setInitialScale(100);
        WebSettings settings = this.web_ads.getSettings();
        settings.setDomStorageEnabled(true);
        settings.setJavaScriptEnabled(true);
        settings.setAllowFileAccess(true);
        settings.setAllowContentAccess(true);
        File file = new File(getCacheDir().getPath() + "/hismith_ads/");
        if (file.exists() && (listFiles = file.listFiles()) != null && listFiles.length > 0) {
            this.showAdsVersion = listFiles[0].getName();
            LogUtil.e("广告版本号：" + this.showAdsVersion);
            Uri fromFile = Uri.fromFile(new File(listFiles[0], "HISMITH_ADS_IMAGE.jpg"));
            LogUtil.e("图片：" + fromFile.toString());
            String readAds = CacheUtil.readAds(this);
            if (!TextUtils.isEmpty(readAds)) {
                String replace = readAds.replace("HISMITH_ADS_IMAGE", fromFile.toString());
                LogUtil.e("Html:" + replace);
                this.web_ads.loadDataWithBaseURL(fromFile.toString(), replace, "text/html", "utf-8", null);
            }
        }
        boolean isEmpty = TextUtils.isEmpty((String) SharedPreferencesUtil.get(this, Constans.Token, ""));
        Handler handler = this.mHandler;
        if (isEmpty) {
            handler.sendEmptyMessageDelayed(0, this.animTime);
        } else {
            handler.sendEmptyMessageDelayed(1, this.animTime);
        }
        this.mHandler.sendEmptyMessage(2);
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        this.mHandler.removeCallbacksAndMessages(null);
    }

    @Override // androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onStop() {
        super.onStop();
        LogUtil.e("ads onStop()");
        String str = (String) SharedPreferencesUtil.get(this, Constans.adsVersion);
        if (TextUtils.isEmpty(str)) {
            CacheUtil.deleteAds(this);
        } else {
            if (this.showAdsVersion.equals(str)) {
                return;
            }
            CacheUtil.deleteAdsVersionOther(this, str);
        }
    }
}
