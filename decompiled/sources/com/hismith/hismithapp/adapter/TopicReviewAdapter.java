package com.hismith.hismithapp.adapter;

import android.graphics.drawable.Drawable;
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
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.dialog.ReviewDetailsDialog;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.ReviewLikeModel;
import com.hismith.hismithapp.model.TopicReviewModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.TimeUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.CircleImageView;
import io.reactivex.Observable;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.List;

/* loaded from: classes3.dex */
public class TopicReviewAdapter extends RecyclerView.Adapter<MyViewHolder> {
    private RetrofitRequest baseRequest = new RetrofitRequest();
    private BaseActivity mContext;
    private List<TopicReviewModel.Review> mList;
    private OnUserInfoListener onUserInfoListener;
    private Integer topicId;

    class MyViewHolder extends RecyclerView.ViewHolder {
        private ImageView mTopicReviewImgDelete;
        private CircleImageView mTopicReviewImgHead;
        private ImageView mTopicReviewImgLike;
        private LinearLayout mTopicReviewLayoutReply;
        private TextView mTopicReviewTxtContent;
        private TextView mTopicReviewTxtDate;
        private TextView mTopicReviewTxtLike;
        private TextView mTopicReviewTxtName;
        private TextView mTopicReviewTxtReply;
        private TextView mTopicReviewTxtReplyLabel;

        public MyViewHolder(View view) {
            super(view);
            this.mTopicReviewImgHead = (CircleImageView) view.findViewById(R.id.topic_review_img_head);
            this.mTopicReviewImgLike = (ImageView) view.findViewById(R.id.topic_review_img_like);
            this.mTopicReviewImgDelete = (ImageView) view.findViewById(R.id.topic_review_img_delete);
            this.mTopicReviewTxtName = (TextView) view.findViewById(R.id.topic_review_txt_name);
            this.mTopicReviewTxtContent = (TextView) view.findViewById(R.id.topic_review_txt_content);
            this.mTopicReviewTxtLike = (TextView) view.findViewById(R.id.topic_review_txt_like);
            this.mTopicReviewTxtDate = (TextView) view.findViewById(R.id.topic_review_txt_date);
            this.mTopicReviewTxtReply = (TextView) view.findViewById(R.id.topic_review_txt_reply);
            this.mTopicReviewTxtReplyLabel = (TextView) view.findViewById(R.id.topic_review_txt_reply_label);
            this.mTopicReviewLayoutReply = (LinearLayout) view.findViewById(R.id.topic_review_layout_reply);
            LanguageUtil.setText(TopicReviewAdapter.this.mContext, this.mTopicReviewTxtReplyLabel, "reply");
        }
    }

    private class OnUserInfoClickListener implements View.OnClickListener {
        private int mPosition;

        public OnUserInfoClickListener(int i) {
            this.mPosition = i;
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            if (TopicReviewAdapter.this.onUserInfoListener == null || ((TopicReviewModel.Review) TopicReviewAdapter.this.mList.get(this.mPosition)).getUserId() == null) {
                return;
            }
            TopicReviewAdapter.this.onUserInfoListener.onUserInfoClick(((TopicReviewModel.Review) TopicReviewAdapter.this.mList.get(this.mPosition)).getUserId().intValue());
        }
    }

    public interface OnUserInfoListener {
        void onUserInfoClick(int i);
    }

    public TopicReviewAdapter(BaseActivity baseActivity, Integer num, List<TopicReviewModel.Review> list) {
        this.mContext = baseActivity;
        this.topicId = num;
        this.mList = list;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicReviewDel(final Integer num) {
        Observable<HttpModel> observeOn = this.baseRequest.getApi().topicReviewDel((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), this.mList.get(num.intValue()).getId()).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.mContext;
        observeOn.subscribe(new RxSubscribe<HttpModel>(baseActivity, baseActivity.progressDialog, true) { // from class: com.hismith.hismithapp.adapter.TopicReviewAdapter.6
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    TopicReviewAdapter.this.mList.remove(num.intValue());
                    ToastUtil.showShortToastCenter(TopicReviewAdapter.this.mContext, LanguageUtil.getLanguageByKey(TopicReviewAdapter.this.mContext, "deletedMsg"));
                    TopicReviewAdapter.this.notifyItemRemoved(num.intValue());
                    TopicReviewAdapter.this.notifyItemRangeChanged(num.intValue(), TopicReviewAdapter.this.mList.size());
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicReviewLike(final Integer num, final String str) {
        this.baseRequest.getApi().topicReviewLike((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), this.mList.get(num.intValue()).getId(), str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<ReviewLikeModel>(this.mContext, null, false) { // from class: com.hismith.hismithapp.adapter.TopicReviewAdapter.5
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(ReviewLikeModel reviewLikeModel) {
                if (reviewLikeModel.getSuccess() == 1) {
                    if ("unlike".equals(str)) {
                        ((TopicReviewModel.Review) TopicReviewAdapter.this.mList.get(num.intValue())).setLikeed(0);
                    } else if ("like".equals(str)) {
                        ((TopicReviewModel.Review) TopicReviewAdapter.this.mList.get(num.intValue())).setLikeed(1);
                    }
                    ((TopicReviewModel.Review) TopicReviewAdapter.this.mList.get(num.intValue())).setLikeCount(reviewLikeModel.getLikeCount());
                    TopicReviewAdapter.this.notifyItemChanged(num.intValue(), "");
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
        try {
            Glide.with((FragmentActivity) this.mContext).load(this.mList.get(i).getUserHeadIcon()).placeholder(R.mipmap.ic_personal_head).into((RequestBuilder) new SimpleTarget<Drawable>() { // from class: com.hismith.hismithapp.adapter.TopicReviewAdapter.1
                public void onResourceReady(Drawable drawable, Transition<? super Drawable> transition) {
                    myViewHolder.mTopicReviewImgHead.setImageDrawable(drawable);
                }

                @Override // com.bumptech.glide.request.target.Target
                public /* bridge */ /* synthetic */ void onResourceReady(Object obj, Transition transition) {
                    onResourceReady((Drawable) obj, (Transition<? super Drawable>) transition);
                }
            });
            myViewHolder.mTopicReviewImgHead.setKing(this.mList.get(i).getUserKing().intValue(), this.mContext, this.mList.get(i).getGradeIcon());
            myViewHolder.mTopicReviewTxtName.setText(this.mList.get(i).getUserName());
            myViewHolder.mTopicReviewTxtContent.setText(this.mList.get(i).getContent());
            myViewHolder.mTopicReviewTxtLike.setText(this.mList.get(i).getLikeCount() + "");
            myViewHolder.mTopicReviewTxtDate.setText(TimeUtil.utc2Local(this.mList.get(i).getCreateDate()));
            myViewHolder.mTopicReviewTxtReply.setText(this.mList.get(i).getReplyCount() + "");
            if (MyApplication.getApplication().getmUserModel() == null || MyApplication.getApplication().getmUserModel().getUserId() != this.mList.get(i).getUserId().intValue()) {
                myViewHolder.mTopicReviewImgDelete.setVisibility(8);
            } else {
                myViewHolder.mTopicReviewImgDelete.setVisibility(0);
            }
            if (this.mList.get(i).getLikeed().intValue() == 1) {
                myViewHolder.mTopicReviewImgLike.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_like_pressed));
                myViewHolder.mTopicReviewTxtLike.setTextColor(this.mContext.getResources().getColor(R.color.item_review_liked_text_color));
            } else {
                myViewHolder.mTopicReviewImgLike.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_like));
                myViewHolder.mTopicReviewTxtLike.setTextColor(this.mContext.getResources().getColor(R.color.item_review_like_text_color));
            }
            OnUserInfoClickListener onUserInfoClickListener = new OnUserInfoClickListener(i);
            myViewHolder.mTopicReviewImgHead.setOnClickListener(onUserInfoClickListener);
            myViewHolder.mTopicReviewTxtName.setOnClickListener(onUserInfoClickListener);
            myViewHolder.mTopicReviewImgLike.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.TopicReviewAdapter.2
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    TopicReviewAdapter.this.topicReviewLike(Integer.valueOf(i), ((TopicReviewModel.Review) TopicReviewAdapter.this.mList.get(i)).getLikeed().intValue() == 1 ? "unlike" : "like");
                }
            });
            myViewHolder.mTopicReviewLayoutReply.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.TopicReviewAdapter.3
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    new ReviewDetailsDialog(TopicReviewAdapter.this.mContext, ((TopicReviewModel.Review) TopicReviewAdapter.this.mList.get(i)).getId()).showDialog();
                }
            });
            myViewHolder.mTopicReviewImgDelete.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.TopicReviewAdapter.4
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    LogUtil.e("删除评论ID：" + ((TopicReviewModel.Review) TopicReviewAdapter.this.mList.get(i)).getId());
                    TopicReviewAdapter.this.topicReviewDel(Integer.valueOf(i));
                }
            });
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* renamed from: onBindViewHolder, reason: avoid collision after fix types in other method */
    public void onBindViewHolder2(MyViewHolder myViewHolder, int i, List<Object> list) {
        try {
            if (list.isEmpty()) {
                onBindViewHolder(myViewHolder, i);
                return;
            }
            myViewHolder.mTopicReviewTxtLike.setText(this.mList.get(i).getLikeCount() + "");
            if (this.mList.get(i).getLikeed().intValue() == 1) {
                myViewHolder.mTopicReviewImgLike.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_like_pressed));
                myViewHolder.mTopicReviewTxtLike.setTextColor(this.mContext.getResources().getColor(R.color.item_review_liked_text_color));
            } else {
                myViewHolder.mTopicReviewImgLike.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_like));
                myViewHolder.mTopicReviewTxtLike.setTextColor(this.mContext.getResources().getColor(R.color.item_review_like_text_color));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public MyViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new MyViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_topic_review, viewGroup, false));
    }

    public void setOnUserInfoListener(OnUserInfoListener onUserInfoListener) {
        this.onUserInfoListener = onUserInfoListener;
    }
}
