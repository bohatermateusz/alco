package com.hismith.hismithapp.dialog;

import android.content.Context;
import android.text.TextUtils;
import android.view.View;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;
import androidx.fragment.app.FragmentManager;
import androidx.media3.extractor.text.ttml.TtmlNode;
import com.bumptech.glide.Glide;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.databinding.DialogPlayVideoPanelBinding;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.PlayPanelView;
import com.lxj.xpopup.XPopup;
import com.lxj.xpopup.core.BasePopupView;
import com.lxj.xpopup.core.BottomPopupView;
import com.lxj.xpopup.interfaces.SimpleCallback;

/* loaded from: classes3.dex */
public class PlayVideoPanelDialog extends BottomPopupView {
    private boolean funs1Control;
    private boolean funs2Control;
    private DialogPlayVideoPanelBinding mBinding;
    private int mCheckTrack;
    private Context mContext;
    private int mFuns2SafeMax;
    private BasePopupView mPlayVideoPopupView;
    private int mSafeMax;
    private onDialogStatusListener onDialogStatusListener;
    private onSafeChangeListener onSafeChangeListener;

    public interface onDialogStatusListener {
        void beforeDismiss();

        void beforeShow(int i);
    }

    public interface onSafeChangeListener {
        void onFuns2Control(boolean z);

        void onFuns2Safe(int i);

        void onSpeedControl(boolean z);

        void onSpeedSafe(int i);

        void onTrackCheckChange(int i);
    }

    public PlayVideoPanelDialog(Context context) {
        super(context);
        this.mSafeMax = 100;
        this.mFuns2SafeMax = 100;
        this.funs1Control = true;
        this.funs2Control = true;
        this.mCheckTrack = 273;
        this.mContext = context;
    }

    public PlayVideoPanelDialog(Context context, int i, int i2, boolean z, boolean z2) {
        super(context);
        this.mCheckTrack = 273;
        this.mContext = context;
        this.mSafeMax = i;
        this.mFuns2SafeMax = i2;
        this.funs1Control = z;
        this.funs2Control = z2;
    }

    private void addPlayPanelFragment() {
        FragmentManager supportFragmentManager = ((FragmentActivity) this.mContext).getSupportFragmentManager();
        if (((PlayPanelView) supportFragmentManager.findFragmentById(R.id.layout_device_panel)) == null) {
            PlayPanelView playPanelView = new PlayPanelView(this.mSafeMax, this.mFuns2SafeMax);
            playPanelView.setOnSafeChangeListener(new PlayPanelView.onSafeChangeListener() { // from class: com.hismith.hismithapp.dialog.PlayVideoPanelDialog.2
                @Override // com.hismith.hismithapp.view.PlayPanelView.onSafeChangeListener
                public void onFuns2Safe(int i) {
                    if (PlayVideoPanelDialog.this.onSafeChangeListener != null) {
                        PlayVideoPanelDialog.this.onSafeChangeListener.onFuns2Safe(i);
                    }
                }

                @Override // com.hismith.hismithapp.view.PlayPanelView.onSafeChangeListener
                public void onProductInfo() {
                    PlayVideoPanelDialog.this.initMotorUI();
                }

                @Override // com.hismith.hismithapp.view.PlayPanelView.onSafeChangeListener
                public void onSpeedSafe(int i) {
                    if (PlayVideoPanelDialog.this.onSafeChangeListener != null) {
                        PlayVideoPanelDialog.this.onSafeChangeListener.onSpeedSafe(i);
                    }
                }
            });
            supportFragmentManager.beginTransaction().replace(R.id.layout_device_panel, playPanelView).commitAllowingStateLoss();
        }
    }

    private void initData() {
        this.mBinding.layoutRoot.setPadding(this.mBinding.layoutRoot.getPaddingStart(), this.mBinding.layoutRoot.getPaddingTop(), this.mBinding.layoutRoot.getPaddingEnd(), this.mBinding.layoutRoot.getPaddingBottom() + MeasureUtil.getNavigationBarHeight(this.mContext));
        this.mCheckTrack = ((Integer) SharedPreferencesUtil.get(this.mContext, Constans.VIDEO_TRACK_CHECK, 273)).intValue();
        updateTrackUI();
        initMotorUI();
        addPlayPanelFragment();
        setAfterApplySize(new Runnable() { // from class: com.hismith.hismithapp.dialog.PlayVideoPanelDialog.1
            @Override // java.lang.Runnable
            public void run() {
                if (PlayVideoPanelDialog.this.mBinding == null) {
                    return;
                }
                LogUtil.e("setAfterApplySize height:" + PlayVideoPanelDialog.this.mBinding.layoutRoot.getHeight());
                if (PlayVideoPanelDialog.this.onDialogStatusListener != null) {
                    PlayVideoPanelDialog.this.onDialogStatusListener.beforeShow(PlayVideoPanelDialog.this.mBinding.layoutRoot.getHeight() - MeasureUtil.getNavigationBarHeight(PlayVideoPanelDialog.this.mContext));
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initMotorUI() {
        if (!ProductUtil.isSMVType()) {
            this.mBinding.imgMotor1.setVisibility(8);
            this.mBinding.imgMotor1Select.setVisibility(8);
            this.mBinding.imgMotor2.setVisibility(8);
            this.mBinding.imgMotor2Select.setVisibility(8);
            return;
        }
        Glide.with(getContext()).load(Integer.valueOf(R.mipmap.ic_mode_select)).placeholder(R.mipmap.ic_mode_select).into(this.mBinding.imgMotor1Select);
        boolean z = this.funs1Control;
        DialogPlayVideoPanelBinding dialogPlayVideoPanelBinding = this.mBinding;
        if (z) {
            dialogPlayVideoPanelBinding.imgMotor1Select.setVisibility(0);
        } else {
            dialogPlayVideoPanelBinding.imgMotor1Select.setVisibility(4);
        }
        this.mBinding.imgMotor1.setVisibility(0);
        if (!TextUtils.isEmpty(ProductUtil.getProductFun(0).getfIcon())) {
            Glide.with(this.mContext).load(ProductUtil.getProductFun(0).getfIcon()).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(this.mBinding.imgMotor1);
        } else if (ProductUtil.getProductFun(0).getfIconLocation() > 0) {
            Glide.with(this.mContext).load(Integer.valueOf(ProductUtil.getProductFun(0).getfIconLocation())).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(this.mBinding.imgMotor1);
        }
        Glide.with(getContext()).load(Integer.valueOf(R.mipmap.ic_mode_select)).placeholder(R.mipmap.ic_mode_select).into(this.mBinding.imgMotor2Select);
        boolean z2 = this.funs2Control;
        DialogPlayVideoPanelBinding dialogPlayVideoPanelBinding2 = this.mBinding;
        if (z2) {
            dialogPlayVideoPanelBinding2.imgMotor2Select.setVisibility(0);
        } else {
            dialogPlayVideoPanelBinding2.imgMotor2Select.setVisibility(4);
        }
        this.mBinding.imgMotor2.setVisibility(0);
        if (!TextUtils.isEmpty(ProductUtil.getProductFun(1).getfIcon())) {
            Glide.with(this.mContext).load(ProductUtil.getProductFun(1).getfIcon()).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(this.mBinding.imgMotor2);
        } else if (ProductUtil.getProductFun(1).getfIconLocation() > 0) {
            Glide.with(this.mContext).load(Integer.valueOf(ProductUtil.getProductFun(1).getfIconLocation())).circleCrop().placeholder(R.drawable.bg_motor_circle_black).into(this.mBinding.imgMotor2);
        }
    }

    private void removePlaySMSFragment() {
        FragmentManager supportFragmentManager = ((FragmentActivity) this.mContext).getSupportFragmentManager();
        Fragment findFragmentById = supportFragmentManager.findFragmentById(R.id.layout_device_panel);
        if (findFragmentById != null) {
            supportFragmentManager.beginTransaction().remove(findFragmentById).commitAllowingStateLoss();
        }
    }

    private void setListener() {
        this.mBinding.imgClose.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.dialog.PlayVideoPanelDialog$$ExternalSyntheticLambda0
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                PlayVideoPanelDialog.this.m585xc3790772(view);
            }
        });
        this.mBinding.txtTrackHead.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.dialog.PlayVideoPanelDialog$$ExternalSyntheticLambda1
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                PlayVideoPanelDialog.this.switchCheck(view);
            }
        });
        this.mBinding.txtTrackHands.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.dialog.PlayVideoPanelDialog$$ExternalSyntheticLambda1
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                PlayVideoPanelDialog.this.switchCheck(view);
            }
        });
        this.mBinding.txtTrackHips.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.dialog.PlayVideoPanelDialog$$ExternalSyntheticLambda1
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                PlayVideoPanelDialog.this.switchCheck(view);
            }
        });
        this.mBinding.imgMotor1.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.dialog.PlayVideoPanelDialog$$ExternalSyntheticLambda2
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                PlayVideoPanelDialog.this.m586xf151a1d1(view);
            }
        });
        this.mBinding.imgMotor2.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.dialog.PlayVideoPanelDialog$$ExternalSyntheticLambda3
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                PlayVideoPanelDialog.this.m587x1f2a3c30(view);
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void switchCheck(View view) {
        int i;
        int id = view.getId();
        if (id == R.id.txt_track_head) {
            i = 1;
        } else if (id == R.id.txt_track_hands) {
            i = 16;
        } else if (id != R.id.txt_track_hips) {
            return;
        } else {
            i = 256;
        }
        int i2 = i ^ this.mCheckTrack;
        if ((i2 & 273) == 0) {
            ToastUtil.showWarnShortToast(this.mContext, "selectAtLeastOnePart");
            return;
        }
        this.mCheckTrack = i2;
        SharedPreferencesUtil.put(this.mContext, Constans.VIDEO_TRACK_CHECK, Integer.valueOf(i2));
        updateTrackUI();
        onSafeChangeListener onsafechangelistener = this.onSafeChangeListener;
        if (onsafechangelistener != null) {
            onsafechangelistener.onTrackCheckChange(this.mCheckTrack);
        }
    }

    private void updateTrackUI() {
        this.mBinding.txtTrackHead.setBackgroundResource((this.mCheckTrack & 1) != 0 ? R.drawable.bg_track_select : R.drawable.bg_track_normal);
        this.mBinding.txtTrackHands.setBackgroundResource((this.mCheckTrack & 16) != 0 ? R.drawable.bg_track_select : R.drawable.bg_track_normal);
        this.mBinding.txtTrackHips.setBackgroundResource((this.mCheckTrack & 256) != 0 ? R.drawable.bg_track_select : R.drawable.bg_track_normal);
    }

    public void dissDialog() {
        BasePopupView basePopupView = this.mPlayVideoPopupView;
        if (basePopupView == null || !basePopupView.isShow()) {
            return;
        }
        this.mPlayVideoPopupView.dismiss();
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected void doAfterShow() {
        super.doAfterShow();
    }

    @Override // com.lxj.xpopup.core.BottomPopupView, com.lxj.xpopup.core.BasePopupView
    protected int getImplLayoutId() {
        return R.layout.dialog_play_video_panel;
    }

    /* renamed from: lambda$setListener$0$com-hismith-hismithapp-dialog-PlayVideoPanelDialog, reason: not valid java name */
    /* synthetic */ void m585xc3790772(View view) {
        dissDialog();
    }

    /* renamed from: lambda$setListener$1$com-hismith-hismithapp-dialog-PlayVideoPanelDialog, reason: not valid java name */
    /* synthetic */ void m586xf151a1d1(View view) {
        boolean z = this.funs1Control;
        this.funs1Control = !z;
        DialogPlayVideoPanelBinding dialogPlayVideoPanelBinding = this.mBinding;
        if (z) {
            dialogPlayVideoPanelBinding.imgMotor1Select.setVisibility(4);
        } else {
            dialogPlayVideoPanelBinding.imgMotor1Select.setVisibility(0);
        }
        onSafeChangeListener onsafechangelistener = this.onSafeChangeListener;
        if (onsafechangelistener != null) {
            onsafechangelistener.onSpeedControl(this.funs1Control);
        }
    }

    /* renamed from: lambda$setListener$2$com-hismith-hismithapp-dialog-PlayVideoPanelDialog, reason: not valid java name */
    /* synthetic */ void m587x1f2a3c30(View view) {
        boolean z = this.funs2Control;
        this.funs2Control = !z;
        DialogPlayVideoPanelBinding dialogPlayVideoPanelBinding = this.mBinding;
        if (z) {
            dialogPlayVideoPanelBinding.imgMotor2Select.setVisibility(4);
        } else {
            dialogPlayVideoPanelBinding.imgMotor2Select.setVisibility(0);
        }
        onSafeChangeListener onsafechangelistener = this.onSafeChangeListener;
        if (onsafechangelistener != null) {
            onsafechangelistener.onFuns2Control(this.funs2Control);
        }
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected void onCreate() {
        super.onCreate();
        DialogPlayVideoPanelBinding bind = DialogPlayVideoPanelBinding.bind(getPopupImplView());
        this.mBinding = bind;
        LanguageUtil.setText(this.mContext, bind.txtTrackTitle, "trackParts");
        LanguageUtil.setText(this.mContext, this.mBinding.txtTrackHead, TtmlNode.TAG_HEAD);
        LanguageUtil.setText(this.mContext, this.mBinding.txtTrackHands, "hand");
        LanguageUtil.setText(this.mContext, this.mBinding.txtTrackHips, "hip");
        initData();
        setListener();
    }

    @Override // com.lxj.xpopup.core.BottomPopupView, com.lxj.xpopup.core.BasePopupView, android.view.ViewGroup, android.view.View
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        this.mBinding = null;
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected void onDismiss() {
        super.onDismiss();
        removePlaySMSFragment();
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected void onShow() {
        super.onShow();
    }

    public void setOnDialogStatusListener(onDialogStatusListener ondialogstatuslistener) {
        this.onDialogStatusListener = ondialogstatuslistener;
    }

    public void setOnSafeChangeListener(onSafeChangeListener onsafechangelistener) {
        this.onSafeChangeListener = onsafechangelistener;
    }

    public void showDialog() {
        BasePopupView asCustom = new XPopup.Builder(this.mContext).hasShadowBg(false).isViewMode(true).setPopupCallback(new SimpleCallback() { // from class: com.hismith.hismithapp.dialog.PlayVideoPanelDialog.3
            @Override // com.lxj.xpopup.interfaces.SimpleCallback, com.lxj.xpopup.interfaces.XPopupCallback
            public void beforeDismiss(BasePopupView basePopupView) {
                super.beforeDismiss(basePopupView);
                if (PlayVideoPanelDialog.this.mBinding == null) {
                    return;
                }
                LogUtil.e("beforeDismiss height:" + PlayVideoPanelDialog.this.mBinding.layoutRoot.getHeight());
                if (PlayVideoPanelDialog.this.onDialogStatusListener != null) {
                    PlayVideoPanelDialog.this.onDialogStatusListener.beforeDismiss();
                }
            }
        }).asCustom(this);
        this.mPlayVideoPopupView = asCustom;
        asCustom.show();
    }
}
