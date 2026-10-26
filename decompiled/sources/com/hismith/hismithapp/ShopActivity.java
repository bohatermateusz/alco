package com.hismith.hismithapp;

import android.app.AlertDialog;
import android.content.DialogInterface;
import android.graphics.Bitmap;
import android.net.http.SslError;
import android.os.Bundle;
import android.view.View;
import android.webkit.JsResult;
import android.webkit.SslErrorHandler;
import android.webkit.WebChromeClient;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.activity.OnBackPressedCallback;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;

/* loaded from: classes3.dex */
public class ShopActivity extends BaseActivity implements View.OnClickListener {
    public static final String EXTRA_WEB_URL = "extra_web_url";
    private ImageView mTitleImgBack;
    private ProgressBar mTitleProgress;
    private TextView mTitleTxtName;
    private WebView mWebView;
    private String url;

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
                if (!str.startsWith("http:") && !str.startsWith("https:")) {
                    AppUtil.startUriToWeb(ShopActivity.this, str);
                    return true;
                }
                webView.loadUrl(str);
            } catch (Exception unused) {
            }
            return true;
        }
    }

    private void init() {
        initWebView();
    }

    private void initView() {
        this.mTitleImgBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleTxtName = (TextView) findViewById(R.id.title_txt_name);
        this.mTitleProgress = (ProgressBar) findViewById(R.id.title_progress);
        this.mWebView = (WebView) findViewById(R.id.shop_webView);
    }

    private void initWebView() {
        this.mWebView.getSettings().setCacheMode(1);
        this.mWebView.setWebChromeClient(new WebChromeClient() { // from class: com.hismith.hismithapp.ShopActivity.2
            @Override // android.webkit.WebChromeClient
            public boolean onJsAlert(WebView webView, String str, String str2, JsResult jsResult) {
                return super.onJsAlert(webView, str, str2, jsResult);
            }

            @Override // android.webkit.WebChromeClient
            public void onProgressChanged(WebView webView, int i) {
                super.onProgressChanged(webView, i);
                ShopActivity.this.mTitleProgress.setProgress(i);
                if (i < 100) {
                    ShopActivity.this.mTitleProgress.setVisibility(0);
                } else if (i == 100) {
                    ShopActivity.this.mTitleProgress.setVisibility(8);
                }
            }
        });
        this.mWebView.setWebViewClient(new GameWebViewClient());
        this.mWebView.getSettings().setJavaScriptEnabled(true);
        LogUtil.e("webView uri:");
        this.mWebView.loadUrl("");
    }

    private void receivedSslErrorDialog(WebView webView, final SslErrorHandler sslErrorHandler, SslError sslError) {
        String languageByKey = LanguageUtil.getLanguageByKey(this, "sslVerificationFailed");
        String languageByKey2 = LanguageUtil.getLanguageByKey(this, "buttomProceed");
        String languageByKey3 = LanguageUtil.getLanguageByKey(this, "cancel");
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setMessage(languageByKey);
        builder.setPositiveButton(languageByKey2, new DialogInterface.OnClickListener() { // from class: com.hismith.hismithapp.ShopActivity.3
            @Override // android.content.DialogInterface.OnClickListener
            public void onClick(DialogInterface dialogInterface, int i) {
                sslErrorHandler.proceed();
            }
        });
        builder.setNegativeButton(languageByKey3, new DialogInterface.OnClickListener() { // from class: com.hismith.hismithapp.ShopActivity.4
            @Override // android.content.DialogInterface.OnClickListener
            public void onClick(DialogInterface dialogInterface, int i) {
                sslErrorHandler.cancel();
            }
        });
        builder.setOnCancelListener(new DialogInterface.OnCancelListener() { // from class: com.hismith.hismithapp.ShopActivity.5
            @Override // android.content.DialogInterface.OnCancelListener
            public void onCancel(DialogInterface dialogInterface) {
                sslErrorHandler.cancel();
            }
        });
        builder.create().show();
    }

    private void setListener() {
        this.mTitleImgBack.setOnClickListener(this);
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() == R.id.title_img_back) {
            finish();
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) { // from class: com.hismith.hismithapp.ShopActivity.1
            @Override // androidx.activity.OnBackPressedCallback
            public void handleOnBackPressed() {
                boolean canGoBack = ShopActivity.this.mWebView.canGoBack();
                ShopActivity shopActivity = ShopActivity.this;
                if (canGoBack) {
                    shopActivity.mWebView.goBack();
                } else {
                    shopActivity.finish();
                }
            }
        });
        setContentView(R.layout.activity_shop);
        initView();
        init();
        setListener();
    }
}
