package com.hismith.hismithapp.dialog;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.text.TextUtils;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.RemotePlayActivity;
import com.hismith.hismithapp.adapter.ShareAdapter;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.manager.WxShareUtils;
import com.hismith.hismithapp.model.ShareModel;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.lxj.xpopup.XPopup;
import com.lxj.xpopup.core.BottomPopupView;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class ShareDialog extends BottomPopupView implements View.OnClickListener {
    public static final int ID_COPYLINE = 5;
    public static final int ID_FACEBOOK = 1;
    public static final int ID_FACEBOOK_MESSENGER = 7;
    public static final int ID_FRIEND = 6;
    public static final int ID_LINE = 2;
    public static final int ID_WECHAT = 4;
    public static final int ID_WHATSAPP = 3;
    private boolean isClose;
    private Activity mActivity;
    private ShareAdapter mAdapter;
    private ImageView mImgClose;
    private GridLayoutManager mLayoutManager;
    private List<ShareModel> mList;
    private RecyclerView mRecycler;
    private String mRemoteUrl;
    private TextView mTxtName;

    public ShareDialog(Activity activity) {
        super(activity);
        this.mRemoteUrl = "";
        this.isClose = true;
        this.mActivity = activity;
    }

    public ShareDialog(Activity activity, String str) {
        super(activity);
        this.isClose = true;
        this.mActivity = activity;
        this.mRemoteUrl = str;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public Boolean isShareAppInstall(String str) {
        try {
            this.mActivity.getPackageManager().getApplicationInfo(str, 128);
            return true;
        } catch (PackageManager.NameNotFoundException unused) {
            return false;
        }
    }

    @Override // com.lxj.xpopup.core.BottomPopupView, com.lxj.xpopup.core.BasePopupView
    protected int getImplLayoutId() {
        return R.layout.dialog_link_share;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        dismiss();
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected void onCreate() {
        super.onCreate();
        this.mTxtName = (TextView) findViewById(R.id.dialog_share_txt_name);
        this.mImgClose = (ImageView) findViewById(R.id.dialog_share_img_close);
        this.mRecycler = (RecyclerView) findViewById(R.id.dialog_share_recycler);
        LanguageUtil.setText(this.mActivity, this.mTxtName, "inviteFriendsControl");
        GridLayoutManager gridLayoutManager = new GridLayoutManager(this.mActivity, 4);
        this.mLayoutManager = gridLayoutManager;
        this.mRecycler.setLayoutManager(gridLayoutManager);
        this.mList = new ArrayList();
        if (!TextUtils.isEmpty(this.mActivity.getString(R.string.facebook_scheme))) {
            this.mList.add(new ShareModel(R.mipmap.ic_share_facebook, this.mActivity.getResources().getString(R.string.android_app_facebook), 1));
        }
        this.mList.add(new ShareModel(R.mipmap.ic_share_line, this.mActivity.getResources().getString(R.string.android_app_line), 2));
        this.mList.add(new ShareModel(R.mipmap.ic_share_whatsapp, this.mActivity.getResources().getString(R.string.android_app_whatsapp), 3));
        if (WxShareUtils.isWXAppInstalled(this.mActivity) && !TextUtils.isEmpty(this.mActivity.getString(R.string.wx_id))) {
            this.mList.add(new ShareModel(R.mipmap.ic_share_wechat, this.mActivity.getResources().getString(R.string.android_app_wechat), 4));
        }
        if (!TextUtils.isEmpty(this.mActivity.getString(R.string.facebook_scheme))) {
            this.mList.add(new ShareModel(R.mipmap.ic_share_messenger, this.mActivity.getResources().getString(R.string.android_app_messenger), 7));
        }
        this.mList.add(new ShareModel(R.mipmap.ic_share_firend, LanguageUtil.getLanguageByKey(this.mActivity, "friendCount"), 6));
        this.mList.add(new ShareModel(R.mipmap.ic_share_copylink, LanguageUtil.getLanguageByKey(this.mActivity, "copyLink"), 5));
        ShareAdapter shareAdapter = new ShareAdapter(this.mActivity, this.mList);
        this.mAdapter = shareAdapter;
        this.mRecycler.setAdapter(shareAdapter);
        this.mImgClose.setOnClickListener(this);
        this.mAdapter.setOnItemCLickListener(new ShareAdapter.OnItemCLickListener() { // from class: com.hismith.hismithapp.dialog.ShareDialog.1
            @Override // com.hismith.hismithapp.adapter.ShareAdapter.OnItemCLickListener
            public void onItemClick(int i) {
                switch (i) {
                    case 1:
                        Intent intent = new Intent(ShareDialog.this.mActivity, (Class<?>) RemotePlayActivity.class);
                        intent.putExtra("shareId", 1);
                        intent.putExtra("remoteUrl", ShareDialog.this.mRemoteUrl);
                        ShareDialog.this.mActivity.startActivity(intent);
                        break;
                    case 2:
                        if (!ShareDialog.this.isShareAppInstall("jp.naver.line.android").booleanValue()) {
                            AppUtil.shareAppDl(ShareDialog.this.mActivity, "jp.naver.line.android");
                            break;
                        } else {
                            Intent intent2 = new Intent(ShareDialog.this.mActivity, (Class<?>) RemotePlayActivity.class);
                            intent2.putExtra("shareId", 2);
                            intent2.putExtra("remoteUrl", ShareDialog.this.mRemoteUrl);
                            ShareDialog.this.mActivity.startActivity(intent2);
                            break;
                        }
                    case 3:
                        if (!ShareDialog.this.isShareAppInstall("com.whatsapp").booleanValue()) {
                            AppUtil.shareAppDl(ShareDialog.this.mActivity, "com.whatsapp");
                            break;
                        } else {
                            Intent intent3 = new Intent(ShareDialog.this.mActivity, (Class<?>) RemotePlayActivity.class);
                            intent3.putExtra("shareId", 3);
                            intent3.putExtra("remoteUrl", ShareDialog.this.mRemoteUrl);
                            ShareDialog.this.mActivity.startActivity(intent3);
                            break;
                        }
                    case 4:
                        Intent intent4 = new Intent(ShareDialog.this.mActivity, (Class<?>) RemotePlayActivity.class);
                        intent4.putExtra("shareId", 4);
                        intent4.putExtra("remoteUrl", ShareDialog.this.mRemoteUrl);
                        ShareDialog.this.mActivity.startActivity(intent4);
                        break;
                    case 5:
                        Intent intent5 = new Intent(ShareDialog.this.mActivity, (Class<?>) RemotePlayActivity.class);
                        intent5.putExtra("shareId", 5);
                        intent5.putExtra("remoteUrl", ShareDialog.this.mRemoteUrl);
                        ShareDialog.this.mActivity.startActivity(intent5);
                        break;
                    case 6:
                        Intent intent6 = new Intent(ShareDialog.this.mActivity, (Class<?>) RemotePlayActivity.class);
                        intent6.putExtra("isShowShare", true);
                        intent6.putExtra("shareId", 6);
                        intent6.putExtra("remoteUrl", ShareDialog.this.mRemoteUrl);
                        ShareDialog.this.mActivity.startActivity(intent6);
                        break;
                    case 7:
                        Intent intent7 = new Intent(ShareDialog.this.mActivity, (Class<?>) RemotePlayActivity.class);
                        intent7.putExtra("shareId", 7);
                        intent7.putExtra("remoteUrl", ShareDialog.this.mRemoteUrl);
                        ShareDialog.this.mActivity.startActivity(intent7);
                        break;
                }
                ShareDialog.this.isClose = false;
                ShareDialog.this.dismiss();
            }
        });
    }

    @Override // com.lxj.xpopup.core.BottomPopupView, com.lxj.xpopup.core.BasePopupView, android.view.ViewGroup, android.view.View
    public void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        if (this.isClose) {
            SocketManager.getInstance().close();
            SocketManager.getInstance().closeSocket();
        }
    }

    public void showDialog() {
        new XPopup.Builder(this.mActivity).asCustom(this).show();
    }
}
