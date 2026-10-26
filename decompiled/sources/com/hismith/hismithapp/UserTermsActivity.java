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
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.activity.OnBackPressedCallback;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.view.CustomWebView;

/* loaded from: classes3.dex */
public class UserTermsActivity extends BaseActivity implements View.OnClickListener {
    private ImageView mTitleImgBack;
    private ProgressBar mTitleProgress;
    private TextView mTitleTxtName;
    private CustomWebView mWebView;

    class GameWebViewClient extends WebViewClient {
        GameWebViewClient() {
        }

        @Override // android.webkit.WebViewClient
        public void onPageFinished(WebView webView, String str) {
        }

        @Override // android.webkit.WebViewClient
        public void onPageStarted(WebView webView, String str, Bitmap bitmap) {
        }

        @Override // android.webkit.WebViewClient
        public boolean shouldOverrideUrlLoading(WebView webView, String str) {
            LogUtil.e("shouldOverrideUrlLoading url:" + str);
            webView.loadUrl(str);
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
        this.mWebView = (CustomWebView) findViewById(R.id.userTerms_webView);
        LanguageUtil.setText(this, this.mTitleTxtName, "aboutusUserTerm");
    }

    private void initWebView() {
        this.mWebView.getSettings().setCacheMode(1);
        this.mWebView.setBackgroundColor(0);
        this.mWebView.setWebChromeClient(new WebChromeClient() { // from class: com.hismith.hismithapp.UserTermsActivity.2
            @Override // android.webkit.WebChromeClient
            public boolean onJsAlert(WebView webView, String str, String str2, JsResult jsResult) {
                return super.onJsAlert(webView, str, str2, jsResult);
            }

            @Override // android.webkit.WebChromeClient
            public void onProgressChanged(WebView webView, int i) {
                super.onProgressChanged(webView, i);
                UserTermsActivity.this.mTitleProgress.setProgress(i);
                if (i < 100) {
                    UserTermsActivity.this.mTitleProgress.setVisibility(0);
                } else if (i == 100) {
                    UserTermsActivity.this.mTitleProgress.setVisibility(8);
                }
            }
        });
        this.mWebView.setWebViewClient(new GameWebViewClient());
        WebSettings settings = this.mWebView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setAllowFileAccess(true);
        settings.setAllowContentAccess(true);
        String str = "https://www.hismiths.com/cms/user-terms?app&language=" + LanguageUtil.getCurrLanguage(this);
        LogUtil.e("webView uri:" + str);
        this.mWebView.loadUrl(str);
    }

    private void receivedSslErrorDialog(WebView webView, final SslErrorHandler sslErrorHandler, SslError sslError) {
        String languageByKey = LanguageUtil.getLanguageByKey(this, "sslVerificationFailed");
        String languageByKey2 = LanguageUtil.getLanguageByKey(this, "buttomProceed");
        String languageByKey3 = LanguageUtil.getLanguageByKey(this, "cancel");
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setMessage(languageByKey);
        builder.setPositiveButton(languageByKey2, new DialogInterface.OnClickListener() { // from class: com.hismith.hismithapp.UserTermsActivity.3
            @Override // android.content.DialogInterface.OnClickListener
            public void onClick(DialogInterface dialogInterface, int i) {
                sslErrorHandler.proceed();
            }
        });
        builder.setNegativeButton(languageByKey3, new DialogInterface.OnClickListener() { // from class: com.hismith.hismithapp.UserTermsActivity.4
            @Override // android.content.DialogInterface.OnClickListener
            public void onClick(DialogInterface dialogInterface, int i) {
                sslErrorHandler.cancel();
            }
        });
        builder.setOnCancelListener(new DialogInterface.OnCancelListener() { // from class: com.hismith.hismithapp.UserTermsActivity.5
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
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) { // from class: com.hismith.hismithapp.UserTermsActivity.1
            @Override // androidx.activity.OnBackPressedCallback
            public void handleOnBackPressed() {
                boolean canGoBack = UserTermsActivity.this.mWebView.canGoBack();
                UserTermsActivity userTermsActivity = UserTermsActivity.this;
                if (canGoBack) {
                    userTermsActivity.mWebView.goBack();
                } else {
                    userTermsActivity.finish();
                }
            }
        });
        setContentView(R.layout.activity_user_terms);
        initView();
        init();
        setListener();
    }
}
