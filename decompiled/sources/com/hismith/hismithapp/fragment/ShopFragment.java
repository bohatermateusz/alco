package com.hismith.hismithapp.fragment;

import android.app.AlertDialog;
import android.content.DialogInterface;
import android.graphics.Bitmap;
import android.net.http.SslError;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.webkit.JsResult;
import android.webkit.SslErrorHandler;
import android.webkit.WebChromeClient;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.MainActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.StatusBarUtil;
import com.hismith.hismithapp.view.ObservableWebView;

/* loaded from: classes3.dex */
public class ShopFragment extends LazyLoadFragment {
    private BaseActivity baseActivity;
    private LinearLayout mLayoutRoot;
    private ProgressBar mTitleProgress;
    private View mTitleView;
    private ObservableWebView mWebView;
    private int topDistance = 0;
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
                    AppUtil.startUriToWeb(ShopFragment.this.baseActivity, str);
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

    private void initWebView() {
        this.mWebView.getSettings().setCacheMode(1);
        this.mWebView.setWebChromeClient(new WebChromeClient() { // from class: com.hismith.hismithapp.fragment.ShopFragment.3
            @Override // android.webkit.WebChromeClient
            public boolean onJsAlert(WebView webView, String str, String str2, JsResult jsResult) {
                return super.onJsAlert(webView, str, str2, jsResult);
            }

            @Override // android.webkit.WebChromeClient
            public void onProgressChanged(WebView webView, int i) {
                super.onProgressChanged(webView, i);
                if (i < 100) {
                    ShopFragment.this.mTitleProgress.setVisibility(0);
                } else if (i == 100) {
                    ShopFragment.this.mTitleProgress.setVisibility(8);
                }
            }
        });
        this.mWebView.setWebViewClient(new GameWebViewClient());
        this.mWebView.getSettings().setJavaScriptEnabled(true);
        String str = ((String) SharedPreferencesUtil.get(this.baseActivity, Constans.storeUrl)) + "?token=" + ((String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token));
        LogUtil.e("url:" + str);
        this.mWebView.loadUrl(str);
    }

    private void receivedSslErrorDialog(WebView webView, final SslErrorHandler sslErrorHandler, SslError sslError) {
        String languageByKey = LanguageUtil.getLanguageByKey(this.baseActivity, "sslVerificationFailed");
        String languageByKey2 = LanguageUtil.getLanguageByKey(this.baseActivity, "buttomProceed");
        String languageByKey3 = LanguageUtil.getLanguageByKey(this.baseActivity, "cancel");
        AlertDialog.Builder builder = new AlertDialog.Builder(this.baseActivity);
        builder.setMessage(languageByKey);
        builder.setPositiveButton(languageByKey2, new DialogInterface.OnClickListener() { // from class: com.hismith.hismithapp.fragment.ShopFragment.4
            @Override // android.content.DialogInterface.OnClickListener
            public void onClick(DialogInterface dialogInterface, int i) {
                sslErrorHandler.proceed();
            }
        });
        builder.setNegativeButton(languageByKey3, new DialogInterface.OnClickListener() { // from class: com.hismith.hismithapp.fragment.ShopFragment.5
            @Override // android.content.DialogInterface.OnClickListener
            public void onClick(DialogInterface dialogInterface, int i) {
                sslErrorHandler.cancel();
            }
        });
        builder.setOnCancelListener(new DialogInterface.OnCancelListener() { // from class: com.hismith.hismithapp.fragment.ShopFragment.6
            @Override // android.content.DialogInterface.OnCancelListener
            public void onCancel(DialogInterface dialogInterface) {
                sslErrorHandler.cancel();
            }
        });
        builder.create().show();
    }

    private void setListener() {
        this.mWebView.setOnTouchListener(new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.ShopFragment.1
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                if (motionEvent.getAction() != 0) {
                    return false;
                }
                LogUtil.e("ACTION_DOWN");
                ((MainActivity) ShopFragment.this.baseActivity).changeMenuHeight(ShopFragment.this.mWebView.getScrollY());
                return false;
            }
        });
        this.mWebView.setOnScrollChangedCallback(new ObservableWebView.OnScrollChangedCallback() { // from class: com.hismith.hismithapp.fragment.ShopFragment.2
            @Override // com.hismith.hismithapp.view.ObservableWebView.OnScrollChangedCallback
            public void onScroll(int i, int i2) {
                LogUtil.e("onScroll num1:" + (ShopFragment.this.mWebView.getContentHeight() * ShopFragment.this.mWebView.getScale()) + ", num2:" + (ShopFragment.this.mWebView.getHeight() + (MeasureUtil.dip2px(ShopFragment.this.baseActivity, 60.0f) * 2)));
                if (ShopFragment.this.mWebView.getContentHeight() * ShopFragment.this.mWebView.getScale() > ShopFragment.this.mWebView.getHeight() + (MeasureUtil.dip2px(ShopFragment.this.baseActivity, 60.0f) * 2)) {
                    ((MainActivity) ShopFragment.this.baseActivity).changeMenuHeight(ShopFragment.this.mWebView.getScrollY());
                }
            }
        });
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        init();
        setListener();
    }

    public boolean onBackPressed() {
        ObservableWebView observableWebView = this.mWebView;
        if (observableWebView == null || !observableWebView.canGoBack()) {
            return false;
        }
        this.mWebView.goBack();
        return true;
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.baseActivity = (BaseActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.activity_shop, (ViewGroup) null);
        this.mLayoutRoot = (LinearLayout) inflate.findViewById(R.id.layout_root);
        this.mTitleView = inflate.findViewById(R.id.title_view);
        this.mTitleProgress = (ProgressBar) inflate.findViewById(R.id.title_progress);
        this.mWebView = (ObservableWebView) inflate.findViewById(R.id.shop_webView);
        StatusBarUtil.fitTitleBar(this.baseActivity, this.mTitleView);
        return inflate;
    }

    @Override // androidx.fragment.app.Fragment
    public void onPause() {
        super.onPause();
        ObservableWebView observableWebView = this.mWebView;
        if (observableWebView != null) {
            observableWebView.onPause();
        }
    }

    @Override // androidx.fragment.app.Fragment
    public void onResume() {
        super.onResume();
        ObservableWebView observableWebView = this.mWebView;
        if (observableWebView != null) {
            observableWebView.onResume();
        }
    }
}
