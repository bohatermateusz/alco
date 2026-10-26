package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.content.Intent;
import android.graphics.drawable.Drawable;
import android.text.SpannableString;
import android.text.method.LinkMovementMethod;
import android.text.style.ClickableSpan;
import android.text.style.ForegroundColorSpan;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.request.target.SimpleTarget;
import com.bumptech.glide.request.transition.Transition;
import com.hismith.hismithapp.PersonalDetailsActivity2;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.UserDetailsActivity2;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.ReviewDetailsModel;
import com.hismith.hismithapp.model.ReviewLikeModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.TimeUtil;
import com.hismith.hismithapp.view.CircleImageView;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.List;

/* loaded from: classes3.dex */
public class ReviewDetailsAdapter extends RecyclerView.Adapter<MyViewHolder> {
    private RetrofitRequest baseRequest = new RetrofitRequest();
    private Context mContext;
    private List<ReviewDetailsModel.Review> mList;
    private OnItemReplyClickListener onItemReplyClickListener;
    private ReviewDelClickListener reviewDelClickListener;

    class MyViewHolder extends RecyclerView.ViewHolder {
        private ImageView mImgDelete;
        private CircleImageView mImgHead;
        private ImageView mImgLike;
        private TextView mTxtContent;
        private TextView mTxtLike;
        private TextView mTxtName;
        private TextView mTxtReply;
        private TextView mTxtTime;

        public MyViewHolder(View view) {
            super(view);
            this.mImgHead = (CircleImageView) view.findViewById(R.id.item_review_details_img_head);
            this.mImgLike = (ImageView) view.findViewById(R.id.item_review_details_img_like);
            this.mImgDelete = (ImageView) view.findViewById(R.id.item_review_details_img_delete);
            this.mTxtName = (TextView) view.findViewById(R.id.item_review_details_txt_name);
            this.mTxtTime = (TextView) view.findViewById(R.id.item_review_details_txt_time);
            this.mTxtLike = (TextView) view.findViewById(R.id.item_review_details_txt_like);
            this.mTxtContent = (TextView) view.findViewById(R.id.item_review_details_txt_content);
            this.mTxtReply = (TextView) view.findViewById(R.id.item_review_details_txt_reply);
            LanguageUtil.setText(ReviewDetailsAdapter.this.mContext, this.mTxtReply, "reply");
        }
    }

    public interface OnItemReplyClickListener {
        void itemReplyClick(int i);
    }

    private class OnUserInfoClickListener implements View.OnClickListener {
        private int mPosition;

        public OnUserInfoClickListener(int i) {
            this.mPosition = i;
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            if (((ReviewDetailsModel.Review) ReviewDetailsAdapter.this.mList.get(this.mPosition)).getUserId() != null) {
                if (MyApplication.getApplication().getmUserModel() != null && ((ReviewDetailsModel.Review) ReviewDetailsAdapter.this.mList.get(this.mPosition)).getUserId().intValue() == MyApplication.getApplication().getmUserModel().getUserId()) {
                    ReviewDetailsAdapter.this.mContext.startActivity(new Intent(ReviewDetailsAdapter.this.mContext, (Class<?>) PersonalDetailsActivity2.class));
                } else {
                    Intent intent = new Intent(ReviewDetailsAdapter.this.mContext, (Class<?>) UserDetailsActivity2.class);
                    intent.putExtra("userId", ((ReviewDetailsModel.Review) ReviewDetailsAdapter.this.mList.get(this.mPosition)).getUserId());
                    ReviewDetailsAdapter.this.mContext.startActivity(intent);
                }
            }
        }
    }

    public interface ReviewDelClickListener {
        void onReviewDelClick(int i);
    }

    public ReviewDetailsAdapter(Context context, List<ReviewDetailsModel.Review> list) {
        this.mContext = context;
        this.mList = list;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicReviewLike(final Integer num, Integer num2, final String str) {
        this.baseRequest.getApi().topicReviewLike((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), num2, str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<ReviewLikeModel>(this.mContext, null, false) { // from class: com.hismith.hismithapp.adapter.ReviewDetailsAdapter.6
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(ReviewLikeModel reviewLikeModel) {
                if (reviewLikeModel.getSuccess() == 1) {
                    if ("unlike".equals(str)) {
                        ((ReviewDetailsModel.Review) ReviewDetailsAdapter.this.mList.get(num.intValue())).setLikeed(0);
                    } else if ("like".equals(str)) {
                        ((ReviewDetailsModel.Review) ReviewDetailsAdapter.this.mList.get(num.intValue())).setLikeed(1);
                    }
                    ((ReviewDetailsModel.Review) ReviewDetailsAdapter.this.mList.get(num.intValue())).setLikeCount(reviewLikeModel.getLikeCount());
                    ReviewDetailsAdapter.this.notifyItemChanged(num.intValue(), "");
                }
            }
        });
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mList.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public /* bridge */ /* synthetic */ void onBindViewHolder(MyViewHolder myViewHolder, int i, List list) {
        onBindViewHolder2(myViewHolder, i, (List<Object>) list);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(final MyViewHolder myViewHolder, final int i) {
        Glide.with(this.mContext).load(this.mList.get(i).getUserHeadIcon()).into((RequestBuilder<Drawable>) new SimpleTarget<Drawable>() { // from class: com.hismith.hismithapp.adapter.ReviewDetailsAdapter.1
            public void onResourceReady(Drawable drawable, Transition<? super Drawable> transition) {
                myViewHolder.mImgHead.setImageDrawable(drawable);
            }

            @Override // com.bumptech.glide.request.target.Target
            public /* bridge */ /* synthetic */ void onResourceReady(Object obj, Transition transition) {
                onResourceReady((Drawable) obj, (Transition<? super Drawable>) transition);
            }
        });
        myViewHolder.mImgHead.setKing(this.mList.get(i).getUserKing().intValue(), this.mContext, this.mList.get(i).getGradeIcon());
        myViewHolder.mTxtName.setText(this.mList.get(i).getUserName());
        myViewHolder.mTxtTime.setText(TimeUtil.utc2Local(this.mList.get(i).getCreateDate()));
        myViewHolder.mTxtContent.setText(this.mList.get(i).getContent());
        if (this.mList.get(i).getReply() != null) {
            SpannableString spannableString = new SpannableString("@" + (this.mList.get(i).getReply().getUserName() != null ? this.mList.get(i).getReply().getUserName() : ""));
            spannableString.setSpan(new ClickableSpan() { // from class: com.hismith.hismithapp.adapter.ReviewDetailsAdapter.2
                @Override // android.text.style.ClickableSpan
                public void onClick(View view) {
                    LogUtil.e("点击文本响应position：" + i + ", id:" + ((ReviewDetailsModel.Review) ReviewDetailsAdapter.this.mList.get(i)).getReply().getUserId() + ", username:" + ((ReviewDetailsModel.Review) ReviewDetailsAdapter.this.mList.get(i)).getReply().getUserName());
                }
            }, 0, spannableString.length(), 33);
            spannableString.setSpan(new ForegroundColorSpan(this.mContext.getResources().getColor(R.color.community_cancel_txt_color)), 0, spannableString.length(), 33);
            myViewHolder.mTxtContent.append("//");
            myViewHolder.mTxtContent.append(spannableString);
            myViewHolder.mTxtContent.append(" " + this.mList.get(i).getReply().getContent());
            myViewHolder.mTxtContent.setMovementMethod(LinkMovementMethod.getInstance());
        }
        myViewHolder.mTxtLike.setText(this.mList.get(i).getLikeCount() + "");
        if (this.mList.get(i).getUserId().equals(Integer.valueOf(MyApplication.getApplication().getmUserModel().getUserId()))) {
            myViewHolder.mImgDelete.setVisibility(0);
        } else {
            myViewHolder.mImgDelete.setVisibility(8);
        }
        if (this.mList.get(i).getLikeed().intValue() == 1) {
            myViewHolder.mImgLike.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_like_pressed));
        } else {
            myViewHolder.mImgLike.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_like));
        }
        OnUserInfoClickListener onUserInfoClickListener = new OnUserInfoClickListener(i);
        myViewHolder.mImgHead.setOnClickListener(onUserInfoClickListener);
        myViewHolder.mTxtName.setOnClickListener(onUserInfoClickListener);
        myViewHolder.mImgLike.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.ReviewDetailsAdapter.3
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                ReviewDetailsAdapter.this.topicReviewLike(Integer.valueOf(i), ((ReviewDetailsModel.Review) ReviewDetailsAdapter.this.mList.get(i)).getId(), ((ReviewDetailsModel.Review) ReviewDetailsAdapter.this.mList.get(i)).getLikeed().intValue() == 1 ? "unlike" : "like");
            }
        });
        myViewHolder.mTxtReply.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.ReviewDetailsAdapter.4
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (ReviewDetailsAdapter.this.onItemReplyClickListener != null) {
                    ReviewDetailsAdapter.this.onItemReplyClickListener.itemReplyClick(i);
                }
            }
        });
        myViewHolder.mImgDelete.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.ReviewDetailsAdapter.5
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (ReviewDetailsAdapter.this.reviewDelClickListener != null) {
                    ReviewDetailsAdapter.this.reviewDelClickListener.onReviewDelClick(i);
                }
                LogUtil.e("删除评论ID:" + ((ReviewDetailsModel.Review) ReviewDetailsAdapter.this.mList.get(i)).getId());
            }
        });
    }

    /* renamed from: onBindViewHolder, reason: avoid collision after fix types in other method */
    public void onBindViewHolder2(MyViewHolder myViewHolder, int i, List<Object> list) {
        if (list.isEmpty()) {
            onBindViewHolder(myViewHolder, i);
            return;
        }
        myViewHolder.mTxtLike.setText(this.mList.get(i).getLikeCount() + "");
        if (this.mList.get(i).getLikeed().intValue() == 1) {
            myViewHolder.mImgLike.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_like_pressed));
        } else {
            myViewHolder.mImgLike.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_like));
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public MyViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new MyViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_review_details, viewGroup, false));
    }

    public void setOnItemReplyClickListener(OnItemReplyClickListener onItemReplyClickListener) {
        this.onItemReplyClickListener = onItemReplyClickListener;
    }

    public void setReviewDelClickListener(ReviewDelClickListener reviewDelClickListener) {
        this.reviewDelClickListener = reviewDelClickListener;
    }
}
