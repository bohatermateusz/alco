package com.hismith.hismithapp;

import android.content.Context;
import android.content.Intent;
import android.content.res.Resources;
import android.os.Bundle;
import android.os.PersistableBundle;
import android.text.TextUtils;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.core.motion.utils.TypedValues;
import androidx.core.view.WindowInsetsControllerCompat;
import com.google.gson.Gson;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.dialog.ProgressDialog;
import com.hismith.hismithapp.utils.AppManager;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import io.reactivex.disposables.CompositeDisposable;
import org.json.JSONException;
import org.json.JSONObject;

/* loaded from: classes3.dex */
public class BaseActivity extends AppCompatActivity {
    public AppManager appManager = AppManager.getInstance();
    private String contextString;
    public Gson gson;
    public CompositeDisposable mCompositeDisposable;
    public ProgressDialog progressDialog;

    private String getLanguageByKey(String str) {
        JSONObject languageJson = MyApplication.getApplication().getLanguageJson();
        String str2 = "";
        if (languageJson != null) {
            try {
                str2 = languageJson.getJSONObject("langs").getString(str);
            } catch (JSONException e) {
                e.printStackTrace();
            } catch (Exception e2) {
                e2.printStackTrace();
            }
        }
        if (!TextUtils.isEmpty(str2)) {
            return str2;
        }
        try {
            return getResources().getString(getResId("android_" + str));
        } catch (Resources.NotFoundException e3) {
            e3.printStackTrace();
            return str;
        }
    }

    private int getResId(String str) {
        return getResources().getIdentifier(str, TypedValues.Custom.S_STRING, getPackageName());
    }

    private void setHint(View view, String str) {
        if (view instanceof EditText) {
            ((EditText) view).setHint(getLanguageByKey(str));
        }
    }

    private void setText(View view, String str) {
        if (view instanceof TextView) {
            ((TextView) view).setText(getLanguageByKey(str));
        }
    }

    @Override // androidx.appcompat.app.AppCompatActivity, android.app.Activity, android.view.ContextThemeWrapper, android.content.ContextWrapper
    protected void attachBaseContext(Context context) {
        super.attachBaseContext(LanguageUtil.attachBaseContext(context, LanguageUtil.getCurrLanguage(context)));
    }

    protected int getStatusBarColor() {
        return R.color.info_title_color3;
    }

    protected boolean isUseBlackFontWithStatusBar() {
        return getResources().getBoolean(R.bool.UseBlackFontWithStatusBar);
    }

    protected boolean isUseFullScreenMode() {
        return getResources().getBoolean(R.bool.UseBlackFontWithStatusBar);
    }

    public void jumpActivity(Class<?> cls) {
        startActivity(new Intent(this, cls));
    }

    @Override // androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        EdgeToEdge.enable(this);
        this.mCompositeDisposable = new CompositeDisposable();
        this.progressDialog = new ProgressDialog(this, R.style.MyDialog);
        this.gson = new Gson();
        this.appManager.addActivity(this);
        setStatusBar();
    }

    @Override // androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        CompositeDisposable compositeDisposable = this.mCompositeDisposable;
        if (compositeDisposable != null) {
            compositeDisposable.clear();
            this.mCompositeDisposable = null;
        }
        ProgressDialog progressDialog = this.progressDialog;
        if (progressDialog != null) {
            progressDialog.dismiss();
            this.progressDialog = null;
        }
        this.appManager.removeActivity(this);
    }

    @Override // androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onPause() {
        super.onPause();
        LogUtil.e("onPause()");
    }

    @Override // android.app.Activity
    protected void onRestoreInstanceState(Bundle bundle) {
        super.onRestoreInstanceState(bundle);
        LogUtil.e("onRestoreInstanceState(@NonNull Bundle savedInstanceState)");
    }

    @Override // android.app.Activity
    public void onRestoreInstanceState(Bundle bundle, PersistableBundle persistableBundle) {
        super.onRestoreInstanceState(bundle, persistableBundle);
        LogUtil.e("onRestoreInstanceState(@Nullable Bundle savedInstanceState, @Nullable PersistableBundle persistentState)");
    }

    @Override // androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onResume() {
        super.onResume();
        LogUtil.e("onResume()");
    }

    @Override // androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onSaveInstanceState(Bundle bundle) {
        super.onSaveInstanceState(bundle);
        LogUtil.e("onSaveInstanceState(@NonNull @NotNull Bundle outState)");
    }

    @Override // android.app.Activity
    public void onSaveInstanceState(Bundle bundle, PersistableBundle persistableBundle) {
        super.onSaveInstanceState(bundle, persistableBundle);
        LogUtil.e("onSaveInstanceState(@NonNull Bundle outState, @NonNull PersistableBundle outPersistentState)");
    }

    protected void setStatusBar() {
        WindowInsetsControllerCompat windowInsetsControllerCompat = new WindowInsetsControllerCompat(getWindow(), getWindow().getDecorView());
        if (isUseFullScreenMode()) {
            windowInsetsControllerCompat.setAppearanceLightStatusBars(true);
            windowInsetsControllerCompat.setAppearanceLightNavigationBars(true);
        } else {
            windowInsetsControllerCompat.setAppearanceLightStatusBars(false);
            windowInsetsControllerCompat.setAppearanceLightNavigationBars(false);
        }
    }
}
