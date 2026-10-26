package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.content.Intent;
import android.graphics.drawable.Drawable;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.request.target.SimpleTarget;
import com.bumptech.glide.request.transition.Transition;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.model.FollowModel;
import com.hismith.hismithapp.model.socket.SocketUserModel;
import com.hismith.hismithapp.utils.AnimUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.view.CircleImageView;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.List;

/* loaded from: classes3.dex */
public class LineUserAdapter extends RecyclerView.Adapter<ViewHolder> {
    private RetrofitRequest baseRequest = new RetrofitRequest();
    private Context mContext;
    private int mHallUserFlag;
    private List<SocketUserModel> mList;
    private OnItemClickListener onItemClickListener;

    public interface OnItemClickListener {
        void onItemClick(int i);
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private ImageView mImgProgress;
        private ImageView mImgRemoveUser;
        private ImageView mImgUserCaptain;
        private ImageView mImgUserCrown;
        private ImageView mImgUserGender;
        private CircleImageView mImgUserHead;
        private FrameLayout mLayoutRemoveUser;
        private FrameLayout mLayoutUserFollow;
        private TextView mTxtUserFollow;
        private TextView mTxtUserName;

        public ViewHolder(View view) {
            super(view);
            this.mImgUserHead = (CircleImageView) view.findViewById(R.id.item_line_img_user_head);
            this.mTxtUserName = (TextView) view.findViewById(R.id.item_line_txt_user_name);
            this.mTxtUserFollow = (TextView) view.findViewById(R.id.item_line_txt_user_follow);
            this.mImgUserGender = (ImageView) view.findViewById(R.id.item_line_img_user_gender);
            this.mImgUserCrown = (ImageView) view.findViewById(R.id.item_line_img_user_crown);
            this.mImgUserCaptain = (ImageView) view.findViewById(R.id.item_line_img_user_captain);
            this.mLayoutUserFollow = (FrameLayout) view.findViewById(R.id.item_line_layout_user_follow);
            this.mImgProgress = (ImageView) view.findViewById(R.id.item_line_img_progress);
            this.mImgRemoveUser = (ImageView) view.findViewById(R.id.item_line_img_remove_user);
            this.mLayoutRemoveUser = (FrameLayout) view.findViewById(R.id.item_line_layout_remove_user);
        }
    }

    public LineUserAdapter(Context context, List<SocketUserModel> list, int i) {
        this.mContext = context;
        this.mList = list;
        this.mHallUserFlag = i;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void follow(int i, final int i2) {
        String str = (String) SharedPreferencesUtil.get(this.mContext, Constans.Token);
        final int userId = this.mList.get(i2).getUserId();
        this.baseRequest.getApi().follow(str, Integer.valueOf(userId), Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<FollowModel>(this.mContext, null, false) { // from class: com.hismith.hismithapp.adapter.LineUserAdapter.5
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                LineUserAdapter.this.initFollowStatus(userId, 0);
                LineUserAdapter.this.notifyItemChanged(i2, "");
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                LineUserAdapter.this.initFollowStatus(userId, 0);
                LineUserAdapter.this.notifyItemChanged(i2, "");
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(FollowModel followModel) {
                if (followModel.getSuccess() == 1) {
                    ((SocketUserModel) LineUserAdapter.this.mList.get(i2)).setIsFollow(followModel.getFollow().intValue());
                    if (followModel.getFollow().intValue() == 1) {
                        if (MyApplication.getApplication().getmUserModel() != null) {
                            MyApplication.getApplication().getmUserModel().setFocusCount(MyApplication.getApplication().getmUserModel().getFocusCount() + 1);
                            LineUserAdapter.this.mContext.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
                            return;
                        }
                        return;
                    }
                    if (MyApplication.getApplication().getmUserModel() != null) {
                        MyApplication.getApplication().getmUserModel().setFocusCount(MyApplication.getApplication().getmUserModel().getFocusCount() - 1);
                        LineUserAdapter.this.mContext.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
                    }
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initFollowStatus(int i, int i2) {
        for (int i3 = 0; i3 < this.mList.size(); i3++) {
            if (this.mList.get(i3).getUserId() == i) {
                this.mList.get(i3).setFollowStatus(i2);
            }
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
        Glide.with(this.mContext).load(this.mList.get(i).getUserHeadIcon()).into((RequestBuilder<Drawable>) new SimpleTarget<Drawable>() { // from class: com.hismith.hismithapp.adapter.LineUserAdapter.1
            public void onResourceReady(Drawable drawable, Transition<? super Drawable> transition) {
                viewHolder.mImgUserHead.setImageDrawable(drawable);
            }

            @Override // com.bumptech.glide.request.target.Target
            public /* bridge */ /* synthetic */ void onResourceReady(Object obj, Transition transition) {
                onResourceReady((Drawable) obj, (Transition<? super Drawable>) transition);
            }
        });
        viewHolder.mTxtUserName.setText(this.mList.get(i).getUserName());
        if (this.mList.get(i).getUserGender() == 2) {
            viewHolder.mImgUserGender.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_details_transgender));
        } else if (this.mList.get(i).getUserGender() == 1) {
            viewHolder.mImgUserGender.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_details_men));
        } else {
            viewHolder.mImgUserGender.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_details_women));
        }
        if (this.mList.get(i).getCaptain() == 1) {
            viewHolder.mImgUserCaptain.setVisibility(0);
        } else {
            viewHolder.mImgUserCaptain.setVisibility(8);
        }
        if (MyApplication.getApplication().getmUserModel() != null && SocketManager.getInstance().getRoomInfoModel() != null && (MyApplication.getApplication().getmUserModel().getUserId() + "").equals(SocketManager.getInstance().getRoomInfoModel().getUserId())) {
            viewHolder.mLayoutRemoveUser.setVisibility(0);
        }
        if (this.mList.get(i).getKing() == 1) {
            viewHolder.mImgUserCrown.setVisibility(0);
            viewHolder.mLayoutRemoveUser.setVisibility(8);
        } else {
            viewHolder.mImgUserCrown.setVisibility(8);
        }
        if (this.mHallUserFlag == 1) {
            viewHolder.mLayoutRemoveUser.setVisibility(8);
        }
        if (MyApplication.getApplication().getmUserModel() == null || MyApplication.getApplication().getmUserModel().getUserId() != this.mList.get(i).getUserId()) {
            viewHolder.mLayoutUserFollow.setVisibility(0);
        } else {
            viewHolder.mLayoutUserFollow.setVisibility(8);
        }
        int isFollow = this.mList.get(i).getIsFollow();
        Context context = this.mContext;
        if (isFollow == 1) {
            LanguageUtil.setText(context, viewHolder.mTxtUserFollow, "followed");
            viewHolder.mLayoutUserFollow.setBackground(this.mContext.getResources().getDrawable(R.drawable.bg_corners_border_style_15));
            viewHolder.mTxtUserFollow.setTextColor(this.mContext.getResources().getColor(R.color.app_style_color));
        } else {
            viewHolder.mTxtUserFollow.setText("+ " + LanguageUtil.getLanguageByKey(context, "followCount"));
            viewHolder.mLayoutUserFollow.setBackground(this.mContext.getResources().getDrawable(R.drawable.bg_edit_style_15));
            viewHolder.mTxtUserFollow.setTextColor(this.mContext.getResources().getColor(R.color.white));
        }
        if (this.mList.get(i).getFollowStatus() == 1) {
            viewHolder.mLayoutUserFollow.setEnabled(false);
            viewHolder.mLayoutUserFollow.setBackground(this.mContext.getResources().getDrawable(R.drawable.bg_corners_border_style_15));
            viewHolder.mTxtUserFollow.setVisibility(4);
            viewHolder.mImgProgress.setVisibility(0);
            AnimUtil.rotate(viewHolder.mImgProgress);
        } else {
            viewHolder.mLayoutUserFollow.setEnabled(true);
            viewHolder.mTxtUserFollow.setVisibility(0);
            viewHolder.mImgProgress.clearAnimation();
            viewHolder.mImgProgress.setVisibility(4);
        }
        viewHolder.mLayoutUserFollow.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.LineUserAdapter.2
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (i < LineUserAdapter.this.mList.size()) {
                    viewHolder.mLayoutUserFollow.setEnabled(false);
                    viewHolder.mLayoutUserFollow.setBackground(LineUserAdapter.this.mContext.getResources().getDrawable(R.drawable.bg_corners_border_style_15));
                    viewHolder.mTxtUserFollow.setVisibility(4);
                    viewHolder.mImgProgress.setVisibility(0);
                    AnimUtil.rotate(viewHolder.mImgProgress);
                    int i2 = ((SocketUserModel) LineUserAdapter.this.mList.get(i)).getIsFollow() != 1 ? 1 : 0;
                    ((SocketUserModel) LineUserAdapter.this.mList.get(i)).setFollowStatus(1);
                    LineUserAdapter.this.follow(i2, i);
                }
            }
        });
        viewHolder.mLayoutRemoveUser.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.LineUserAdapter.3
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                SocketManager.getInstance().removeUserRoom(((SocketUserModel) LineUserAdapter.this.mList.get(i)).getUserId());
            }
        });
        viewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.LineUserAdapter.4
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (LineUserAdapter.this.onItemClickListener != null) {
                    LineUserAdapter.this.onItemClickListener.onItemClick(i);
                }
            }
        });
    }

    /* renamed from: onBindViewHolder, reason: avoid collision after fix types in other method */
    public void onBindViewHolder2(ViewHolder viewHolder, int i, List<Object> list) {
        LogUtil.e("SocketManager onBindViewHolder position:" + i);
        if (list.isEmpty()) {
            onBindViewHolder(viewHolder, i);
            return;
        }
        viewHolder.mLayoutUserFollow.setEnabled(true);
        viewHolder.mTxtUserFollow.setVisibility(0);
        viewHolder.mImgProgress.clearAnimation();
        viewHolder.mImgProgress.setVisibility(4);
        int isFollow = this.mList.get(i).getIsFollow();
        Context context = this.mContext;
        if (isFollow == 1) {
            LanguageUtil.setText(context, viewHolder.mTxtUserFollow, "followed");
            viewHolder.mLayoutUserFollow.setBackground(this.mContext.getResources().getDrawable(R.drawable.bg_corners_border_style_15));
            viewHolder.mTxtUserFollow.setTextColor(this.mContext.getResources().getColor(R.color.app_style_color));
        } else {
            viewHolder.mTxtUserFollow.setText("+ " + LanguageUtil.getLanguageByKey(context, "followCount"));
            viewHolder.mLayoutUserFollow.setBackground(this.mContext.getResources().getDrawable(R.drawable.bg_edit_style_15));
            viewHolder.mTxtUserFollow.setTextColor(this.mContext.getResources().getColor(R.color.white));
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_line_user, viewGroup, false));
    }

    public void setOnItemClickListener(OnItemClickListener onItemClickListener) {
        this.onItemClickListener = onItemClickListener;
    }
}
