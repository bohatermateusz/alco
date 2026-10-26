package com.hismith.hismithapp.wxapi;

import android.content.Intent;
import android.os.Bundle;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.LogUtil;
import com.tencent.mm.opensdk.modelbase.BaseReq;
import com.tencent.mm.opensdk.modelbase.BaseResp;
import com.tencent.mm.opensdk.openapi.IWXAPI;
import com.tencent.mm.opensdk.openapi.IWXAPIEventHandler;
import com.tencent.mm.opensdk.openapi.WXAPIFactory;

/* loaded from: classes3.dex */
public class WXEntryActivity extends BaseActivity implements IWXAPIEventHandler {
    private IWXAPI wxapi;

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_wxentry);
        IWXAPI createWXAPI = WXAPIFactory.createWXAPI(this, getString(R.string.wx_id));
        this.wxapi = createWXAPI;
        createWXAPI.handleIntent(getIntent(), this);
    }

    @Override // androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, android.app.Activity
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        this.wxapi.handleIntent(intent, this);
    }

    @Override // com.tencent.mm.opensdk.openapi.IWXAPIEventHandler
    public void onReq(BaseReq baseReq) {
    }

    @Override // com.tencent.mm.opensdk.openapi.IWXAPIEventHandler
    public void onResp(BaseResp baseResp) {
        if (baseResp.errCode == 0) {
            if (baseResp.getType() != 2) {
                return;
            }
            finish();
        } else {
            if (baseResp.getType() != 2) {
                return;
            }
            LogUtil.e(">>>errCode = " + baseResp.errCode);
            finish();
        }
    }
}
