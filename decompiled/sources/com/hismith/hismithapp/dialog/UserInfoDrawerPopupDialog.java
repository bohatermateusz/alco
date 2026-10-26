package com.hismith.hismithapp.dialog;

import android.content.Context;
import android.graphics.drawable.Drawable;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.request.target.SimpleTarget;
import com.bumptech.glide.request.transition.Transition;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.model.socket.SocketUserModel;
import com.hismith.hismithapp.utils.IntentUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.view.CircleImageView;
import com.lxj.xpopup.core.CenterPopupView;
import com.lxj.xpopup.util.XPopupUtils;

/* loaded from: classes3.dex */
public class UserInfoDrawerPopupDialog extends CenterPopupView implements View.OnClickListener {
    private String mCaptainId;
    private Context mContext;
    private CircleImageView mImgHead;
    private int mInviteFlag;
    private FrameLayout mLayoutCaptain;
    private SocketUserModel mLineRoomUserModel;
    private ProgressBar mProgressCaptain;
    private TextView mTxtCaptain;
    private TextView mTxtFans;
    private TextView mTxtFansCount;
    private TextView mTxtFollow;
    private TextView mTxtFollowCount;
    private TextView mTxtName;

    public UserInfoDrawerPopupDialog(Context context, SocketUserModel socketUserModel, int i, String str) {
        super(context);
        this.mContext = context;
        this.mLineRoomUserModel = socketUserModel;
        this.mInviteFlag = i;
        this.mCaptainId = str;
    }

    private void initCaptainData() {
        this.mLayoutCaptain.setEnabled(true);
        this.mTxtCaptain.setVisibility(0);
        this.mProgressCaptain.setVisibility(8);
        int captain = this.mLineRoomUserModel.getCaptain();
        Context context = this.mContext;
        if (captain == 1) {
            LanguageUtil.setText(context, this.mTxtCaptain, "cancelCaptain");
            this.mLayoutCaptain.setBackground(this.mContext.getResources().getDrawable(R.drawable.bg_corners_border_style_25));
            this.mTxtCaptain.setTextColor(this.mContext.getResources().getColor(R.color.app_style_color));
        } else {
            LanguageUtil.setText(context, this.mTxtCaptain, "appointCaptain");
            this.mLayoutCaptain.setBackground(this.mContext.getResources().getDrawable(R.drawable.bg_corners_style_25));
            this.mTxtCaptain.setTextColor(this.mContext.getResources().getColor(R.color.white));
        }
    }

    private void initData() {
        initUserData();
    }

    private void initInviteData() {
        this.mLayoutCaptain.setEnabled(true);
        this.mTxtCaptain.setVisibility(0);
        this.mProgressCaptain.setVisibility(8);
        LanguageUtil.setText(this.mContext, this.mTxtCaptain, "inviteTeammates");
        this.mLayoutCaptain.setBackground(this.mContext.getResources().getDrawable(R.drawable.bg_corners_style_25));
        this.mTxtCaptain.setTextColor(this.mContext.getResources().getColor(R.color.white));
    }

    private void initUserData() {
        Glide.with(this.mContext).load(this.mLineRoomUserModel.getUserHeadIcon()).into((RequestBuilder<Drawable>) new SimpleTarget<Drawable>() { // from class: com.hismith.hismithapp.dialog.UserInfoDrawerPopupDialog.1
            public void onResourceReady(Drawable drawable, Transition<? super Drawable> transition) {
                UserInfoDrawerPopupDialog.this.mImgHead.setImageDrawable(drawable);
            }

            @Override // com.bumptech.glide.request.target.Target
            public /* bridge */ /* synthetic */ void onResourceReady(Object obj, Transition transition) {
                onResourceReady((Drawable) obj, (Transition<? super Drawable>) transition);
            }
        });
        this.mTxtName.setText(this.mLineRoomUserModel.getUserName());
        this.mTxtFansCount.setText(this.mLineRoomUserModel.getUserFans() + "");
        this.mTxtFollowCount.setText(this.mLineRoomUserModel.getUserFocus() + "");
        int i = this.mInviteFlag;
        if (i == 0) {
            if (MyApplication.getApplication().getmUserModel() == null || !(MyApplication.getApplication().getmUserModel().getUserId() + "").equals(SocketManager.getInstance().getRoomInfoModel().getUserId())) {
                if (MyApplication.getApplication().getmUserModel() != null && this.mCaptainId.equals(MyApplication.getApplication().getmUserModel().getUserId() + "")) {
                    this.mLayoutCaptain.setVisibility(0);
                }
            } else if (this.mLineRoomUserModel.getUserId() != MyApplication.getApplication().getmUserModel().getUserId()) {
                this.mLayoutCaptain.setVisibility(0);
            }
            initCaptainData();
            return;
        }
        if (i == 1) {
            if (MyApplication.getApplication().getmUserModel() != null && (MyApplication.getApplication().getmUserModel().getUserId() + "").equals(SocketManager.getInstance().getRoomInfoModel().getUserId())) {
                this.mLayoutCaptain.setVisibility(0);
            }
            if (this.mLineRoomUserModel.getKing() == 1) {
                this.mLayoutCaptain.setVisibility(8);
            }
            if (MyApplication.getApplication().getmUserModel() != null && this.mLineRoomUserModel.getUserId() == MyApplication.getApplication().getmUserModel().getUserId()) {
                this.mLayoutCaptain.setVisibility(8);
            }
            initInviteData();
        }
    }

    private void initView() {
        this.mImgHead = (CircleImageView) findViewById(R.id.dialog_line_info_img_head);
        this.mTxtName = (TextView) findViewById(R.id.dialog_line_info_txt_name);
        this.mTxtFansCount = (TextView) findViewById(R.id.dialog_line_info_txt_fans_count);
        this.mTxtFans = (TextView) findViewById(R.id.dialog_line_info_txt_fans);
        this.mTxtFollowCount = (TextView) findViewById(R.id.dialog_line_info_txt_follow_count);
        this.mTxtFollow = (TextView) findViewById(R.id.dialog_line_info_txt_follow);
        this.mTxtCaptain = (TextView) findViewById(R.id.dialog_line_info_txt_captain);
        this.mLayoutCaptain = (FrameLayout) findViewById(R.id.dialog_line_info_layout_captain);
        this.mProgressCaptain = (ProgressBar) findViewById(R.id.dialog_line_info_progress_captain);
        LanguageUtil.setText(this.mContext, this.mTxtFans, "fansCount");
        LanguageUtil.setText(this.mContext, this.mTxtFollow, "following");
    }

    private void setListener() {
        this.mImgHead.setOnClickListener(this);
        this.mLayoutCaptain.setOnClickListener(this);
    }

    public void changeCaptainStatus() {
        this.mLineRoomUserModel.setCaptain(this.mLineRoomUserModel.getCaptain() == 1 ? 0 : 1);
        initCaptainData();
        delayDismiss(300L);
    }

    @Override // com.lxj.xpopup.core.CenterPopupView, com.lxj.xpopup.core.BasePopupView
    protected int getImplLayoutId() {
        return R.layout.dialog_line_user_info;
    }

    @Override // com.lxj.xpopup.core.CenterPopupView, com.lxj.xpopup.core.BasePopupView
    protected int getMaxWidth() {
        return (int) (XPopupUtils.getScreenWidth(getContext()) * 0.9f);
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.dialog_line_info_img_head) {
            try {
                IntentUtil.startUserInfoActivity(this.mContext, this.mLineRoomUserModel.getUserId());
                return;
            } catch (Exception e) {
                e.printStackTrace();
                return;
            }
        }
        if (id == R.id.dialog_line_info_layout_captain) {
            this.mLayoutCaptain.setEnabled(false);
            this.mLayoutCaptain.setBackground(this.mContext.getResources().getDrawable(R.drawable.bg_corners_border_style_25));
            this.mTxtCaptain.setVisibility(4);
            this.mProgressCaptain.setVisibility(0);
            if (this.mInviteFlag == 0) {
                SocketManager.getInstance().roomCaptain(this.mLineRoomUserModel.getCaptain() != 1 ? this.mLineRoomUserModel.getUserId() : 0);
            } else {
                SocketManager.getInstance().inviteUserRoom(this.mLineRoomUserModel.getUserId());
                delayDismiss(500L);
            }
        }
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected void onCreate() {
        super.onCreate();
        initView();
        initData();
        setListener();
    }
}
