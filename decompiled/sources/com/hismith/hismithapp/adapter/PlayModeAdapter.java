package com.hismith.hismithapp.adapter;

import android.content.Intent;
import android.graphics.drawable.Drawable;
import android.os.Parcelable;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.fragment.app.FragmentActivity;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.request.target.SimpleTarget;
import com.bumptech.glide.request.transition.Transition;
import com.facebook.internal.NativeProtocol;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.PersonalDetailsActivity2;
import com.hismith.hismithapp.PlayModeCreateActivity;
import com.hismith.hismithapp.PlayModeCreateActivity2;
import com.hismith.hismithapp.PlayModePlayActivity2;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.UserDetailsActivity2;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.dialog.ReportSelectDialog;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.PlayModeActionModel;
import com.hismith.hismithapp.model.PlayModeModel;
import com.hismith.hismithapp.popup.ActionAttachPopup2;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.surface_view.LineDataUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.DialogUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.StringUtils;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.CircleImageView;
import com.hismith.hismithapp.view.LinePointsView;
import io.reactivex.Observable;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes3.dex */
public class PlayModeAdapter extends RecyclerView.Adapter<ViewHolder> {
    private RetrofitRequest baseRequest;
    private BaseActivity mContext;
    private ActionAttachPopup2 mCurrentAttachPopup;
    private LayoutInflater mInflater;
    private List<PlayModeModel> mList;
    private String mType;
    private boolean skipUser;

    class OnCollectClickListener implements View.OnClickListener {
        private int mPosition;

        public OnCollectClickListener(int i) {
            this.mPosition = i;
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            PlayModeAdapter.this.notifyItemChanged(this.mPosition, "popup");
            int collected = ((PlayModeModel) PlayModeAdapter.this.mList.get(this.mPosition)).getCollected();
            PlayModeAdapter playModeAdapter = PlayModeAdapter.this;
            if (collected == 1) {
                playModeAdapter.playModeAction(this.mPosition, "uncollect");
            } else {
                playModeAdapter.playModeAction(this.mPosition, "collect");
            }
        }
    }

    class OnCopyClickListener implements View.OnClickListener {
        private int mPosition;

        public OnCopyClickListener(int i) {
            this.mPosition = i;
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            int status = BleManager.getInstance(PlayModeAdapter.this.mContext).getStatus();
            PlayModeAdapter playModeAdapter = PlayModeAdapter.this;
            if (status != 2) {
                ToastUtil.showShortToastCenter(PlayModeAdapter.this.mContext, LanguageUtil.getLanguageByKey(playModeAdapter.mContext, "noConnectedDevice"));
                return;
            }
            playModeAdapter.notifyItemChanged(this.mPosition, "popup");
            if (TextUtils.isEmpty(((PlayModeModel) PlayModeAdapter.this.mList.get(this.mPosition)).getModeData())) {
                Intent intent = new Intent(PlayModeAdapter.this.mContext, (Class<?>) PlayModeCreateActivity2.class);
                intent.putExtra(NativeProtocol.WEB_DIALOG_ACTION, "copy");
                intent.putExtra("editMode", (Parcelable) PlayModeAdapter.this.mList.get(this.mPosition));
                PlayModeAdapter.this.mContext.startActivityForResult(intent, 17);
                return;
            }
            Intent intent2 = new Intent(PlayModeAdapter.this.mContext, (Class<?>) PlayModeCreateActivity.class);
            intent2.putExtra(NativeProtocol.WEB_DIALOG_ACTION, "copy");
            intent2.putExtra("editMode", (Parcelable) PlayModeAdapter.this.mList.get(this.mPosition));
            PlayModeAdapter.this.mContext.startActivityForResult(intent2, 17);
        }
    }

    class OnDelClickListener implements View.OnClickListener {
        private int mPosition;

        public OnDelClickListener(int i) {
            this.mPosition = i;
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            PlayModeAdapter.this.notifyItemChanged(this.mPosition, "popup");
            DialogUtil.showConfirmDialog(PlayModeAdapter.this.mContext, "prompt", "deleteMode", null, "confirm", new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.PlayModeAdapter.OnDelClickListener.1
                @Override // android.view.View.OnClickListener
                public void onClick(View view2) {
                    PlayModeAdapter.this.playModeAction(OnDelClickListener.this.mPosition, "del");
                    DialogUtil.dismissDialog();
                }
            });
        }
    }

    class OnEditClickListener implements View.OnClickListener {
        private int mPosition;

        public OnEditClickListener(int i) {
            this.mPosition = i;
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            int status = BleManager.getInstance(PlayModeAdapter.this.mContext).getStatus();
            PlayModeAdapter playModeAdapter = PlayModeAdapter.this;
            if (status != 2) {
                ToastUtil.showShortToastCenter(PlayModeAdapter.this.mContext, LanguageUtil.getLanguageByKey(playModeAdapter.mContext, "noConnectedDevice"));
                return;
            }
            playModeAdapter.notifyItemChanged(this.mPosition, "popup");
            if (TextUtils.isEmpty(((PlayModeModel) PlayModeAdapter.this.mList.get(this.mPosition)).getModeData())) {
                Intent intent = new Intent(PlayModeAdapter.this.mContext, (Class<?>) PlayModeCreateActivity2.class);
                intent.putExtra(NativeProtocol.WEB_DIALOG_ACTION, "edit");
                intent.putExtra("editMode", (Parcelable) PlayModeAdapter.this.mList.get(this.mPosition));
                PlayModeAdapter.this.mContext.startActivityForResult(intent, 17);
                return;
            }
            Intent intent2 = new Intent(PlayModeAdapter.this.mContext, (Class<?>) PlayModeCreateActivity.class);
            intent2.putExtra(NativeProtocol.WEB_DIALOG_ACTION, "edit");
            intent2.putExtra("editMode", (Parcelable) PlayModeAdapter.this.mList.get(this.mPosition));
            PlayModeAdapter.this.mContext.startActivityForResult(intent2, 17);
        }
    }

    class OnItemClickListener implements View.OnClickListener {
        private int mPosition;

        public OnItemClickListener(int i) {
            this.mPosition = i;
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            if (BleManager.getInstance(PlayModeAdapter.this.mContext).getStatus() != 2) {
                ToastUtil.showShortToastCenter(PlayModeAdapter.this.mContext, LanguageUtil.getLanguageByKey(PlayModeAdapter.this.mContext, "noConnectedDevice"));
            } else if (!ProductUtil.hasSType()) {
                ToastUtil.showShortToastCenter(PlayModeAdapter.this.mContext, LanguageUtil.getLanguageByKey(PlayModeAdapter.this.mContext, "deviceNotSupport"));
            } else {
                Intent intent = new Intent(PlayModeAdapter.this.mContext, (Class<?>) PlayModePlayActivity2.class);
                intent.putExtra("playModeModel", (Parcelable) PlayModeAdapter.this.mList.get(this.mPosition));
                PlayModeAdapter.this.mContext.startActivityForResult(intent, 17);
            }
        }
    }

    class OnRenameClickListener implements View.OnClickListener {
        private int mPosition;

        public OnRenameClickListener(int i) {
            this.mPosition = i;
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            PlayModeAdapter.this.notifyItemChanged(this.mPosition, "popup");
            DialogUtil.showEditDialog2(PlayModeAdapter.this.mContext, LanguageUtil.getLanguageByKey(PlayModeAdapter.this.mContext, "modeName"), ((PlayModeModel) PlayModeAdapter.this.mList.get(this.mPosition)).getModeName(), null, null, new DialogUtil.OnEditContentCallback() { // from class: com.hismith.hismithapp.adapter.PlayModeAdapter.OnRenameClickListener.1
                @Override // com.hismith.hismithapp.utils.DialogUtil.OnEditContentCallback
                public void onEditContent(String str) {
                    PlayModeAdapter.this.playModeAction(OnRenameClickListener.this.mPosition, "rename", str);
                }
            });
        }
    }

    class OnReportClickListener implements View.OnClickListener {
        private int mPosition;

        public OnReportClickListener(int i) {
            this.mPosition = i;
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            PlayModeAdapter.this.notifyItemChanged(this.mPosition, "popup");
            new ReportSelectDialog(PlayModeAdapter.this.mContext) { // from class: com.hismith.hismithapp.adapter.PlayModeAdapter.OnReportClickListener.1
                @Override // com.hismith.hismithapp.dialog.ReportSelectDialog
                public void btnSubmit(String str, String str2) {
                    PlayModeAdapter.this.complaint(((PlayModeModel) PlayModeAdapter.this.mList.get(OnReportClickListener.this.mPosition)).getModeId(), str, str2);
                }
            }.showDialog();
        }
    }

    class OnShareClickListener implements View.OnClickListener {
        private int mPosition;

        public OnShareClickListener(int i) {
            this.mPosition = i;
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            PlayModeAdapter.this.notifyItemChanged(this.mPosition, "popup");
            DialogUtil.showConfirmDialog(PlayModeAdapter.this.mContext, "prompt", "shareMode", null, "confirm", new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.PlayModeAdapter.OnShareClickListener.1
                @Override // android.view.View.OnClickListener
                public void onClick(View view2) {
                    PlayModeAdapter.this.playModeAction(OnShareClickListener.this.mPosition, "share");
                    DialogUtil.dismissDialog();
                }
            });
        }
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private ImageView img_has_piu;
        private ImageView mImgCollectCount;
        private CircleImageView mImgDevice;
        private CircleImageView mImgHead;
        private ImageView mImgLikeCount;
        private ImageView mImgMore;
        private ImageView mImgPlayCount;
        private ImageView mImgTime;
        private LinearLayout mLayoutSetting;
        private LinePointsView mLinePoints;
        private TextView mTxtCollectCount;
        private TextView mTxtLikeCount;
        private TextView mTxtName;
        private TextView mTxtPlayCount;
        private TextView mTxtTime;
        private TextView txt_setting_position_max;
        private TextView txt_setting_position_min;
        private TextView txt_setting_smooth;

        public ViewHolder(View view) {
            super(view);
            this.mImgDevice = (CircleImageView) view.findViewById(R.id.item_mode_img_device);
            this.mImgHead = (CircleImageView) view.findViewById(R.id.item_mode_img_head);
            this.img_has_piu = (ImageView) view.findViewById(R.id.img_has_piu);
            this.mImgTime = (ImageView) view.findViewById(R.id.item_mode_img_time);
            this.mLinePoints = (LinePointsView) view.findViewById(R.id.item_mode_LinePoints);
            this.mTxtName = (TextView) view.findViewById(R.id.item_mode_txt_name);
            this.mTxtPlayCount = (TextView) view.findViewById(R.id.item_mode_txt_playCount);
            this.mTxtCollectCount = (TextView) view.findViewById(R.id.item_mode_txt_collectCount);
            this.mTxtLikeCount = (TextView) view.findViewById(R.id.item_mode_txt_likeCount);
            this.mTxtTime = (TextView) view.findViewById(R.id.item_mode_txt_time);
            this.mImgPlayCount = (ImageView) view.findViewById(R.id.item_mode_img_playCount);
            this.mImgCollectCount = (ImageView) view.findViewById(R.id.item_mode_img_collectCount);
            this.mImgLikeCount = (ImageView) view.findViewById(R.id.item_mode_img_likeCount);
            this.mImgMore = (ImageView) view.findViewById(R.id.item_mode_img_more);
            this.mLayoutSetting = (LinearLayout) view.findViewById(R.id.layout_setting);
            this.txt_setting_position_min = (TextView) view.findViewById(R.id.txt_setting_position_min);
            this.txt_setting_position_max = (TextView) view.findViewById(R.id.txt_setting_position_max);
            this.txt_setting_smooth = (TextView) view.findViewById(R.id.txt_setting_smooth);
        }
    }

    public PlayModeAdapter(BaseActivity baseActivity, List<PlayModeModel> list, String str) {
        this.skipUser = false;
        this.mContext = baseActivity;
        this.baseRequest = new RetrofitRequest();
        this.mInflater = LayoutInflater.from(baseActivity);
        this.mList = list;
        this.mType = str;
    }

    public PlayModeAdapter(BaseActivity baseActivity, List<PlayModeModel> list, String str, boolean z) {
        this(baseActivity, list, str);
        this.skipUser = z;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void complaint(int i, String str, String str2) {
        Observable<HttpModel> observeOn = this.baseRequest.getApi().complaint((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), Integer.valueOf(i), "playMode", str, str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.mContext;
        observeOn.subscribe(new RxSubscribe<HttpModel>(baseActivity, baseActivity.progressDialog, true) { // from class: com.hismith.hismithapp.adapter.PlayModeAdapter.5
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    ToastUtil.showShortToastCenter(PlayModeAdapter.this.mContext, LanguageUtil.getLanguageByKey(PlayModeAdapter.this.mContext, "reportedSuccess"));
                }
            }
        });
    }

    private List<ActionAttachPopup2.ActionItem2> getActionList(int i) {
        ArrayList arrayList = new ArrayList();
        if (this.mList.get(i) != null) {
            if (this.mList.get(i).getShared() == 1 && MyApplication.getApplication().getmUserModel() != null && this.mList.get(i).getUserId() != MyApplication.getApplication().getmUserModel().getUserId()) {
                if (this.mList.get(i).getCollected() == 1) {
                    arrayList.add(new ActionAttachPopup2.ActionItem2(R.mipmap.ic_play_mode_item_delete, "uncollect"));
                } else {
                    arrayList.add(new ActionAttachPopup2.ActionItem2(R.mipmap.ic_play_mode_item_add, "addCollect"));
                }
            }
            if (this.mList.get(i).getShared() == 0 && MyApplication.getApplication().getmUserModel() != null && this.mList.get(i).getUserId() == MyApplication.getApplication().getmUserModel().getUserId()) {
                arrayList.add(new ActionAttachPopup2.ActionItem2(R.mipmap.ic_play_mode_item_rename, "rename"));
                arrayList.add(new ActionAttachPopup2.ActionItem2(R.mipmap.ic_play_mode_item_edit, "edit"));
                arrayList.add(new ActionAttachPopup2.ActionItem2(R.mipmap.ic_play_mode_item_share, "modeShare"));
            }
            if (this.mList.get(i).getUserKing() == 1 || (MyApplication.getApplication().getmUserModel() != null && this.mList.get(i).getUserId() == MyApplication.getApplication().getmUserModel().getUserId())) {
                arrayList.add(new ActionAttachPopup2.ActionItem2(R.mipmap.ic_play_mode_item_copy, "copy"));
            }
            if (MyApplication.getApplication().getmUserModel() != null && this.mList.get(i).getUserId() == MyApplication.getApplication().getmUserModel().getUserId()) {
                arrayList.add(new ActionAttachPopup2.ActionItem2(R.mipmap.ic_play_mode_delete, "delete"));
            }
            if (this.mList.get(i).getShared() == 1) {
                arrayList.add(new ActionAttachPopup2.ActionItem2(R.mipmap.ic_play_mode_item_report, "report"));
            }
        }
        return arrayList;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void playModeAction(int i, String str) {
        playModeAction(i, str, null);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void playModeAction(final int i, final String str, final String str2) {
        this.baseRequest.getApi().playModeAction((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), this.mList.get(i).getModeId(), str, str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<PlayModeActionModel>(MyApplication.getApplication(), this.mContext.progressDialog, true) { // from class: com.hismith.hismithapp.adapter.PlayModeAdapter.4
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(PlayModeActionModel playModeActionModel) {
                if (playModeActionModel.getSuccess() == 1) {
                    if ("like".equals(str)) {
                        ((PlayModeModel) PlayModeAdapter.this.mList.get(i)).setLiked(1);
                        ((PlayModeModel) PlayModeAdapter.this.mList.get(i)).setLikeCount(playModeActionModel.getLikeCount());
                        PlayModeAdapter.this.notifyItemChanged(i, "");
                        return;
                    }
                    if ("unlike".equals(str)) {
                        ((PlayModeModel) PlayModeAdapter.this.mList.get(i)).setLiked(0);
                        ((PlayModeModel) PlayModeAdapter.this.mList.get(i)).setLikeCount(playModeActionModel.getLikeCount());
                        PlayModeAdapter.this.notifyItemChanged(i, "");
                        return;
                    }
                    if ("collect".equals(str)) {
                        ((PlayModeModel) PlayModeAdapter.this.mList.get(i)).setCollected(1);
                        ((PlayModeModel) PlayModeAdapter.this.mList.get(i)).setCollectCount(playModeActionModel.getCollectCount());
                        PlayModeAdapter.this.notifyItemChanged(i, "");
                        return;
                    }
                    if ("uncollect".equals(str)) {
                        ((PlayModeModel) PlayModeAdapter.this.mList.get(i)).setCollected(0);
                        ((PlayModeModel) PlayModeAdapter.this.mList.get(i)).setCollectCount(playModeActionModel.getCollectCount());
                        PlayModeAdapter.this.notifyItemChanged(i, "");
                        return;
                    }
                    if ("rename".equals(str)) {
                        ((PlayModeModel) PlayModeAdapter.this.mList.get(i)).setModeName(str2);
                        PlayModeAdapter.this.notifyItemChanged(i, "");
                        return;
                    }
                    if ("share".equals(str)) {
                        ((PlayModeModel) PlayModeAdapter.this.mList.get(i)).setShared(1);
                        ToastUtil.showShortToastCenter(PlayModeAdapter.this.mContext, LanguageUtil.getLanguageByKey(PlayModeAdapter.this.mContext, "complete"));
                    } else if ("del".equals(str)) {
                        PlayModeAdapter.this.mList.remove(i);
                        PlayModeAdapter.this.notifyItemRemoved(i);
                        PlayModeAdapter playModeAdapter = PlayModeAdapter.this;
                        playModeAdapter.notifyItemRangeChanged(i, playModeAdapter.mList.size());
                        ToastUtil.showShortToastCenter(PlayModeAdapter.this.mContext, LanguageUtil.getLanguageByKey(PlayModeAdapter.this.mContext, "complete"));
                    }
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void showMorePopup(View view, int i) {
        ActionAttachPopup2 actionAttachPopup2 = this.mCurrentAttachPopup;
        if (actionAttachPopup2 != null) {
            actionAttachPopup2.dissPop();
            this.mCurrentAttachPopup = null;
        }
        ActionAttachPopup2 actionAttachPopup22 = new ActionAttachPopup2(this.mContext, getActionList(i));
        actionAttachPopup22.setCollectClickListener(new OnCollectClickListener(i));
        actionAttachPopup22.setOnReportClickListener(new OnReportClickListener(i));
        actionAttachPopup22.setOnShareClickListener(new OnShareClickListener(i));
        actionAttachPopup22.setOnRenameClickListener(new OnRenameClickListener(i));
        actionAttachPopup22.setOnDelClickListener(new OnDelClickListener(i));
        actionAttachPopup22.setOnEditClickListener(new OnEditClickListener(i));
        actionAttachPopup22.setOnCopyClickListener(new OnCopyClickListener(i));
        actionAttachPopup22.showPop(view, MeasureUtil.dip2px(this.mContext, -10.0f));
        this.mCurrentAttachPopup = actionAttachPopup22;
    }

    public void dissMorePopup() {
        ActionAttachPopup2 actionAttachPopup2 = this.mCurrentAttachPopup;
        if (actionAttachPopup2 != null) {
            actionAttachPopup2.dissPop();
            this.mCurrentAttachPopup = null;
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mList.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public /* bridge */ /* synthetic */ void onBindViewHolder(ViewHolder viewHolder, int i, List list) {
        onBindViewHolder2(viewHolder, i, (List<Object>) list);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(final ViewHolder viewHolder, final int i) {
        Glide.with((FragmentActivity) this.mContext).load(this.mList.get(i).getProductLogo()).into(viewHolder.mImgDevice);
        viewHolder.mTxtName.setText(this.mList.get(i).getModeName() + "");
        try {
            String modePoints = this.mList.get(i).getModePoints();
            String upperCase = Integer.toHexString(Constans.START_TYPE_PIU).toUpperCase();
            if (TextUtils.isEmpty(modePoints) || !modePoints.toUpperCase().contains(upperCase + Constans.START_TYPE_PIU_DATA1)) {
                viewHolder.img_has_piu.setVisibility(8);
            } else {
                viewHolder.img_has_piu.setVisibility(0);
            }
            String[] buildServerToClientData = LineDataUtil.buildServerToClientData(modePoints, this.mList.get(i).getModeId());
            if (TextUtils.isEmpty(buildServerToClientData[0])) {
                viewHolder.mLinePoints.setData("", "");
            } else {
                viewHolder.mLinePoints.setData(buildServerToClientData);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        viewHolder.mTxtPlayCount.setText(StringUtils.getFormatNum(this.mList.get(i).getPlayCount()));
        viewHolder.mTxtCollectCount.setText(StringUtils.getFormatNum(this.mList.get(i).getCollectCount()));
        viewHolder.mTxtLikeCount.setText(StringUtils.getFormatNum(this.mList.get(i).getLikeCount()));
        viewHolder.mTxtTime.setText(this.mList.get(i).getModeTime());
        viewHolder.mImgCollectCount.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_play_mode_item_collect_pressed));
        viewHolder.mImgLikeCount.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_play_mode_item_like_pressed));
        if (this.skipUser) {
            viewHolder.mImgHead.setVisibility(8);
        } else {
            viewHolder.mImgHead.setVisibility(0);
            viewHolder.mImgHead.setKing(this.mList.get(i).getUserKing(), this.mContext, this.mList.get(i).getGradeIcon());
            viewHolder.mImgHead.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_personal_head));
            Glide.with((FragmentActivity) this.mContext).load(this.mList.get(i).getUserHeadIcon()).placeholder(R.mipmap.ic_personal_head).into((RequestBuilder) new SimpleTarget<Drawable>() { // from class: com.hismith.hismithapp.adapter.PlayModeAdapter.1
                public void onResourceReady(Drawable drawable, Transition<? super Drawable> transition) {
                    viewHolder.mImgHead.setImageDrawable(drawable);
                }

                @Override // com.bumptech.glide.request.target.Target
                public /* bridge */ /* synthetic */ void onResourceReady(Object obj, Transition transition) {
                    onResourceReady((Drawable) obj, (Transition<? super Drawable>) transition);
                }
            });
            viewHolder.mImgHead.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.PlayModeAdapter.2
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    if (MyApplication.getApplication().getmUserModel() != null && ((PlayModeModel) PlayModeAdapter.this.mList.get(i)).getUserId() == MyApplication.getApplication().getmUserModel().getUserId()) {
                        PlayModeAdapter.this.mContext.startActivity(new Intent(PlayModeAdapter.this.mContext, (Class<?>) PersonalDetailsActivity2.class));
                    } else {
                        Intent intent = new Intent(PlayModeAdapter.this.mContext, (Class<?>) UserDetailsActivity2.class);
                        intent.putExtra("userId", ((PlayModeModel) PlayModeAdapter.this.mList.get(i)).getUserId());
                        PlayModeAdapter.this.mContext.startActivity(intent);
                    }
                }
            });
        }
        if (TextUtils.isEmpty(this.mList.get(i).getModeSetting())) {
            viewHolder.mLayoutSetting.setVisibility(8);
        } else {
            viewHolder.mLayoutSetting.setVisibility(0);
            this.mList.get(i).initModeSettingsModel();
            viewHolder.txt_setting_position_min.setText(this.mList.get(i).getModeSettingsModel().getStrokeMin() + "");
            viewHolder.txt_setting_position_max.setText(this.mList.get(i).getModeSettingsModel().getStrokeMax() + "");
            viewHolder.txt_setting_smooth.setText(this.mList.get(i).getModeSettingsModel().getSmooth() + "");
        }
        viewHolder.mImgMore.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.PlayModeAdapter.3
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                PlayModeAdapter.this.showMorePopup(view, i);
            }
        });
        viewHolder.itemView.setOnClickListener(new OnItemClickListener(i));
    }

    /* renamed from: onBindViewHolder, reason: avoid collision after fix types in other method */
    public void onBindViewHolder2(ViewHolder viewHolder, int i, List<Object> list) {
        if (list.isEmpty()) {
            onBindViewHolder(viewHolder, i);
            return;
        }
        viewHolder.mTxtName.setText(this.mList.get(i).getModeName() + "");
        viewHolder.mTxtCollectCount.setText(this.mList.get(i).getCollectCount() + "");
        viewHolder.mTxtLikeCount.setText(this.mList.get(i).getLikeCount() + "");
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new ViewHolder(this.mInflater.inflate(R.layout.item_play_mode, viewGroup, false));
    }
}
