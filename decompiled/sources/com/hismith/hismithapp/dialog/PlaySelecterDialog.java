package com.hismith.hismithapp.dialog;

import android.content.Context;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.BuildConfig;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.PlaySelectorAdapter;
import com.hismith.hismithapp.model.PlaySelectorModel;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.lxj.xpopup.XPopup;
import com.lxj.xpopup.core.BasePopupView;
import com.lxj.xpopup.core.BottomPopupView;
import com.lxj.xpopup.util.XPopupUtils;
import java.util.ArrayList;

/* loaded from: classes3.dex */
public class PlaySelecterDialog extends BottomPopupView implements View.OnClickListener {
    private PlaySelectorAdapter mAdapter;
    private Context mContext;
    private ImageView mImgClose;
    private ImageView mImgIPlay;
    private ImageView mImgLinePlay;
    private ImageView mImgModePlay;
    private ImageView mImgRemotePlay;
    private BasePopupView mPlaySelecterPopupView;
    private TextView mTxtIPlay;
    private TextView mTxtLinePlay;
    private TextView mTxtModePlay;
    private TextView mTxtRemotePlay;
    private OnCloseListener onCloseListener;
    private OnItemClickListener onItemClickListener;
    private ConstraintLayout play_img_bg;
    private RecyclerView recycler_view;
    private int rowSpacingDip;

    public interface OnCloseListener {
        void onClose();
    }

    public interface OnItemClickListener {
        void onItemClick(int i);
    }

    public PlaySelecterDialog(Context context) {
        super(context);
        this.rowSpacingDip = 0;
        this.mContext = context;
    }

    public PlaySelecterDialog(Context context, Context context2, OnItemClickListener onItemClickListener) {
        super(context);
        this.rowSpacingDip = 0;
        this.mContext = context2;
        this.onItemClickListener = onItemClickListener;
    }

    private void initView() {
        this.mImgIPlay = (ImageView) findViewById(R.id.play_img_iPlay);
        this.mImgRemotePlay = (ImageView) findViewById(R.id.play_img_remotePlay);
        this.mImgModePlay = (ImageView) findViewById(R.id.play_img_modePlay);
        this.mImgLinePlay = (ImageView) findViewById(R.id.play_img_linePlay);
        this.mImgClose = (ImageView) findViewById(R.id.play_img_close);
        this.mTxtIPlay = (TextView) findViewById(R.id.play_txt_iPlay);
        this.mTxtRemotePlay = (TextView) findViewById(R.id.play_txt_remotePlay);
        this.mTxtModePlay = (TextView) findViewById(R.id.play_txt_modePlay);
        this.mTxtLinePlay = (TextView) findViewById(R.id.play_txt_linePlay);
        LanguageUtil.setText(this.mContext, this.mTxtIPlay, "iPlay");
        LanguageUtil.setText(this.mContext, this.mTxtRemotePlay, "remotePlay");
        LanguageUtil.setText(this.mContext, this.mTxtModePlay, "playMode");
        LanguageUtil.setText(this.mContext, this.mTxtLinePlay, "playOnline");
        this.play_img_bg = (ConstraintLayout) findViewById(R.id.play_img_bg);
        this.mImgClose = (ImageView) findViewById(R.id.play_img_close);
        this.rowSpacingDip = MeasureUtil.dip2px(this.mContext, 15.0f);
        this.play_img_bg.setPadding(0, 0, 0, MeasureUtil.getNavigationBarHeight(this.mContext));
        this.recycler_view = (RecyclerView) findViewById(R.id.recycler_view);
        GridLayoutManager gridLayoutManager = new GridLayoutManager(this.mContext, 2) { // from class: com.hismith.hismithapp.dialog.PlaySelecterDialog.1
            @Override // androidx.recyclerview.widget.LinearLayoutManager, androidx.recyclerview.widget.RecyclerView.LayoutManager
            public boolean canScrollVertically() {
                return false;
            }
        };
        gridLayoutManager.setSpanSizeLookup(new GridLayoutManager.SpanSizeLookup() { // from class: com.hismith.hismithapp.dialog.PlaySelecterDialog.2
            @Override // androidx.recyclerview.widget.GridLayoutManager.SpanSizeLookup
            public int getSpanSize(int i) {
                return i == PlaySelecterDialog.this.mAdapter.getItemCount() - 1 ? 2 : 1;
            }
        });
        this.recycler_view.setLayoutManager(gridLayoutManager);
        ArrayList arrayList = new ArrayList();
        arrayList.add(new PlaySelectorModel("iPlay", R.mipmap.ic_play_iplay, R.drawable.bg_gradient_play_solo));
        arrayList.add(new PlaySelectorModel("remotePlay", R.mipmap.ic_play_remote_play, R.drawable.bg_gradient_play_remote));
        arrayList.add(new PlaySelectorModel("playMode", R.mipmap.ic_play_mode_play, R.drawable.bg_gradient_play_mode));
        arrayList.add(new PlaySelectorModel("playOnline", R.mipmap.ic_play_line_play, R.drawable.bg_gradient_play_group));
        arrayList.add(new PlaySelectorModel("playVideo", R.mipmap.ic_play_video, R.drawable.bg_gradient_play_video));
        PlaySelectorAdapter playSelectorAdapter = new PlaySelectorAdapter(this.mContext, arrayList);
        this.mAdapter = playSelectorAdapter;
        playSelectorAdapter.setRowSpacingDip(this.rowSpacingDip);
        this.mAdapter.setOnItemClickListener(new PlaySelectorAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.dialog.PlaySelecterDialog$$ExternalSyntheticLambda0
            @Override // com.hismith.hismithapp.adapter.PlaySelectorAdapter.OnItemClickListener
            public final void onItemClick(int i) {
                PlaySelecterDialog.this.m583xa17cc539(i);
            }
        });
        this.recycler_view.setAdapter(this.mAdapter);
        setAfterApplySize(new Runnable() { // from class: com.hismith.hismithapp.dialog.PlaySelecterDialog.3
            @Override // java.lang.Runnable
            public void run() {
                int paddingTop = PlaySelecterDialog.this.recycler_view.getPaddingTop();
                int paddingBottom = PlaySelecterDialog.this.recycler_view.getPaddingBottom();
                int itemCount = (PlaySelecterDialog.this.mAdapter.getItemCount() + 1) / 2;
                int height = ((((int) (PlaySelecterDialog.this.recycler_view.getHeight() * 1.0f)) - paddingTop) - paddingBottom) - ((itemCount - 1) * PlaySelecterDialog.this.rowSpacingDip);
                LogUtil.e("recyclerheight:" + height + ", screenHeight:" + MeasureUtil.getScreenHeight(PlaySelecterDialog.this.mContext) + ", navHeidht:" + MeasureUtil.getNavigationBarHeight(PlaySelecterDialog.this.mContext));
                if (itemCount <= 0 || height <= 0) {
                    return;
                }
                PlaySelecterDialog.this.mAdapter.setItemHeight(height / itemCount);
                PlaySelecterDialog.this.mAdapter.notifyDataSetChanged();
            }
        });
    }

    private void setListener() {
        this.mImgIPlay.setOnClickListener(this);
        this.mImgRemotePlay.setOnClickListener(this);
        this.mImgModePlay.setOnClickListener(this);
        this.mImgLinePlay.setOnClickListener(this);
        this.mImgClose.setOnClickListener(this);
    }

    public void dissDialog() {
        BasePopupView basePopupView = this.mPlaySelecterPopupView;
        if (basePopupView == null || !basePopupView.isShow()) {
            return;
        }
        this.mPlaySelecterPopupView.dismiss();
    }

    @Override // com.lxj.xpopup.core.BottomPopupView, com.lxj.xpopup.core.BasePopupView
    protected int getImplLayoutId() {
        return R.layout.dialog_drawer_play_selecter;
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected int getMaxHeight() {
        return BuildConfig.FLAVOR.equals(BuildConfig.FLAVOR) ? (int) (XPopupUtils.getScreenHeight(getContext()) * 0.5f) : ((int) (XPopupUtils.getScreenHeight(getContext()) * 0.4f)) + MeasureUtil.getNavigationBarHeight(this.mContext);
    }

    /* renamed from: lambda$initView$0$com-hismith-hismithapp-dialog-PlaySelecterDialog, reason: not valid java name */
    /* synthetic */ void m583xa17cc539(int i) {
        OnItemClickListener onItemClickListener = this.onItemClickListener;
        if (onItemClickListener != null) {
            onItemClickListener.onItemClick(i);
        }
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        OnCloseListener onCloseListener;
        int id = view.getId();
        if (id == R.id.play_img_iPlay) {
            OnItemClickListener onItemClickListener = this.onItemClickListener;
            if (onItemClickListener != null) {
                onItemClickListener.onItemClick(0);
                return;
            }
            return;
        }
        if (id == R.id.play_img_remotePlay) {
            OnItemClickListener onItemClickListener2 = this.onItemClickListener;
            if (onItemClickListener2 != null) {
                onItemClickListener2.onItemClick(1);
                return;
            }
            return;
        }
        if (id == R.id.play_img_modePlay) {
            OnItemClickListener onItemClickListener3 = this.onItemClickListener;
            if (onItemClickListener3 != null) {
                onItemClickListener3.onItemClick(2);
                return;
            }
            return;
        }
        if (id == R.id.play_img_linePlay) {
            OnItemClickListener onItemClickListener4 = this.onItemClickListener;
            if (onItemClickListener4 != null) {
                onItemClickListener4.onItemClick(3);
                return;
            }
            return;
        }
        if (id != R.id.play_img_close || (onCloseListener = this.onCloseListener) == null) {
            return;
        }
        onCloseListener.onClose();
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected void onCreate() {
        super.onCreate();
        initView();
        setListener();
    }

    public void setOnCloseListener(OnCloseListener onCloseListener) {
        this.onCloseListener = onCloseListener;
    }

    public void setOnItemClickListener(OnItemClickListener onItemClickListener) {
        this.onItemClickListener = onItemClickListener;
    }

    public void showDialog() {
        BasePopupView asCustom = new XPopup.Builder(this.mContext).isLightNavigationBar(true).asCustom(this);
        this.mPlaySelecterPopupView = asCustom;
        asCustom.show();
    }
}
