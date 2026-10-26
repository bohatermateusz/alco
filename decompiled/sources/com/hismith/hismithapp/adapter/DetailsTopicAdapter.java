package com.hismith.hismithapp.adapter;

import android.content.Intent;
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
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.TopicContentActivity;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.image_preview.ImagePreview;
import com.hismith.hismithapp.model.TopicActionModel;
import com.hismith.hismithapp.model.TopicModel;
import com.hismith.hismithapp.utils.AnimUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.DialogUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.StringUtils;
import com.hismith.hismithapp.utils.TimeUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.CircleImageView;
import com.hismith.hismithapp.view.GridLayout;
import io.reactivex.Observable;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.List;

/* loaded from: classes3.dex */
public class DetailsTopicAdapter extends RecyclerView.Adapter<ViewHolder> {
    private RetrofitRequest baseRequest = new RetrofitRequest();
    private BaseActivity mContext;
    private List<TopicModel> mTopicModelList;

    class ViewHolder extends RecyclerView.ViewHolder {
        private GridLayout mTopicGridLayout;
        private ImageView mTopicImgFavorite;
        private CircleImageView mTopicImgHead;
        private ImageView mTopicImgLike;
        private ImageView mTopicImgProgressFavorite;
        private ImageView mTopicImgProgressLike;
        private ImageView mTopicImgRead;
        private ImageView mTopicImgReview;
        private LinearLayout mTopicLayoutFavorite;
        private LinearLayout mTopicLayoutItem;
        private LinearLayout mTopicLayoutLike;
        private TextView mTopicTxtContent;
        private TextView mTopicTxtDelete;
        private TextView mTopicTxtFavorite;
        private TextView mTopicTxtLike;
        private TextView mTopicTxtName;
        private TextView mTopicTxtRead;
        private TextView mTopicTxtReview;
        private TextView mTopicTxtTime;
        private TextView mTopicTxtTitle;

        public ViewHolder(View view) {
            super(view);
            this.mTopicLayoutItem = (LinearLayout) view.findViewById(R.id.details_topic_layout_item);
            this.mTopicImgHead = (CircleImageView) view.findViewById(R.id.details_topic_img_head);
            this.mTopicImgLike = (ImageView) view.findViewById(R.id.details_topic_img_like);
            this.mTopicImgReview = (ImageView) view.findViewById(R.id.details_topic_img_review);
            this.mTopicImgFavorite = (ImageView) view.findViewById(R.id.details_topic_img_favorite);
            this.mTopicImgRead = (ImageView) view.findViewById(R.id.details_topic_img_read);
            this.mTopicTxtName = (TextView) view.findViewById(R.id.details_topic_txt_name);
            this.mTopicTxtTime = (TextView) view.findViewById(R.id.details_topic_txt_time);
            this.mTopicTxtDelete = (TextView) view.findViewById(R.id.details_topic_txt_delete);
            this.mTopicTxtTitle = (TextView) view.findViewById(R.id.details_topic_txt_title);
            this.mTopicTxtContent = (TextView) view.findViewById(R.id.details_topic_txt_content);
            this.mTopicTxtLike = (TextView) view.findViewById(R.id.details_topic_txt_like);
            this.mTopicTxtReview = (TextView) view.findViewById(R.id.details_topic_txt_review);
            this.mTopicTxtFavorite = (TextView) view.findViewById(R.id.details_topic_txt_favorite);
            this.mTopicTxtRead = (TextView) view.findViewById(R.id.details_topic_txt_read);
            this.mTopicGridLayout = (GridLayout) view.findViewById(R.id.details_topic_grid_layout);
            this.mTopicLayoutLike = (LinearLayout) view.findViewById(R.id.details_topic_layout_like);
            this.mTopicLayoutFavorite = (LinearLayout) view.findViewById(R.id.details_topic_layout_favorite);
            this.mTopicImgProgressLike = (ImageView) view.findViewById(R.id.details_topic_img_progress_like);
            this.mTopicImgProgressFavorite = (ImageView) view.findViewById(R.id.details_topic_img_progress_favorite);
            LanguageUtil.setText(DetailsTopicAdapter.this.mContext, this.mTopicTxtDelete, "delete");
        }
    }

    public DetailsTopicAdapter(BaseActivity baseActivity, List<TopicModel> list) {
        this.mContext = baseActivity;
        this.mTopicModelList = list;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicCollectAction(final int i, Integer num, final String str) {
        this.baseRequest.getApi().topicAction((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), num, str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicActionModel>(this.mContext, null, false) { // from class: com.hismith.hismithapp.adapter.DetailsTopicAdapter.8
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                ((TopicModel) DetailsTopicAdapter.this.mTopicModelList.get(i)).setCollectStatus(0);
                DetailsTopicAdapter.this.notifyItemChanged(i, "collect");
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                ((TopicModel) DetailsTopicAdapter.this.mTopicModelList.get(i)).setCollectStatus(0);
                DetailsTopicAdapter.this.notifyItemChanged(i, "collect");
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicActionModel topicActionModel) {
                if (topicActionModel.getSuccess() == 1) {
                    String collectCount = topicActionModel.getCollectCount();
                    if ("collect".equals(str)) {
                        ((TopicModel) DetailsTopicAdapter.this.mTopicModelList.get(i)).setCollected(1);
                    } else if ("uncollect".equals(str)) {
                        ((TopicModel) DetailsTopicAdapter.this.mTopicModelList.get(i)).setCollected(0);
                    }
                    ((TopicModel) DetailsTopicAdapter.this.mTopicModelList.get(i)).setCollectCount(collectCount);
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicDelete(final int i) {
        Observable<HttpModel> observeOn = this.baseRequest.getApi().topicDel((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), this.mTopicModelList.get(i).getId()).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.mContext;
        observeOn.subscribe(new RxSubscribe<HttpModel>(baseActivity, baseActivity.progressDialog) { // from class: com.hismith.hismithapp.adapter.DetailsTopicAdapter.6
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                try {
                    if (httpModel.getSuccess() == 1) {
                        DetailsTopicAdapter.this.mTopicModelList.remove(i);
                        ToastUtil.showShortToastCenter(DetailsTopicAdapter.this.mContext, LanguageUtil.getLanguageByKey(DetailsTopicAdapter.this.mContext, "deletedMsg"));
                        DetailsTopicAdapter.this.notifyItemRemoved(i);
                        DetailsTopicAdapter detailsTopicAdapter = DetailsTopicAdapter.this;
                        detailsTopicAdapter.notifyItemRangeChanged(i, detailsTopicAdapter.mTopicModelList.size());
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicLikeAction(final int i, Integer num, final String str) {
        this.baseRequest.getApi().topicAction((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), num, str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicActionModel>(this.mContext, null, false) { // from class: com.hismith.hismithapp.adapter.DetailsTopicAdapter.7
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                ((TopicModel) DetailsTopicAdapter.this.mTopicModelList.get(i)).setLikeStatus(0);
                DetailsTopicAdapter.this.notifyItemChanged(i, "like");
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                ((TopicModel) DetailsTopicAdapter.this.mTopicModelList.get(i)).setLikeStatus(0);
                DetailsTopicAdapter.this.notifyItemChanged(i, "like");
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicActionModel topicActionModel) {
                if (topicActionModel.getSuccess() == 1) {
                    String likeCount = topicActionModel.getLikeCount();
                    if ("like".equals(str)) {
                        ((TopicModel) DetailsTopicAdapter.this.mTopicModelList.get(i)).setLikeed(1);
                    } else if ("unlike".equals(str)) {
                        ((TopicModel) DetailsTopicAdapter.this.mTopicModelList.get(i)).setLikeed(0);
                    }
                    ((TopicModel) DetailsTopicAdapter.this.mTopicModelList.get(i)).setLikeCount(likeCount);
                }
            }
        });
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mTopicModelList.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public /* bridge */ /* synthetic */ void onBindViewHolder(ViewHolder viewHolder, int i, List list) {
        onBindViewHolder2(viewHolder, i, (List<Object>) list);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(final ViewHolder viewHolder, final int i) {
        viewHolder.mTopicTxtName.setText(this.mTopicModelList.get(i).getUserName());
        viewHolder.mTopicTxtTime.setText(TimeUtil.utc2Local(this.mTopicModelList.get(i).getCreateDate()));
        viewHolder.mTopicTxtTitle.setText(this.mTopicModelList.get(i).getTitle());
        if (TextUtils.isEmpty(this.mTopicModelList.get(i).getContent())) {
            viewHolder.mTopicTxtContent.setVisibility(8);
        } else {
            viewHolder.mTopicTxtContent.setVisibility(0);
            viewHolder.mTopicTxtContent.setText(this.mTopicModelList.get(i).getContent());
        }
        viewHolder.mTopicTxtLike.setText(StringUtils.getFormatNum(this.mTopicModelList.get(i).getLikeCount()));
        viewHolder.mTopicTxtReview.setText(StringUtils.getFormatNum(this.mTopicModelList.get(i).getReviewCount()));
        viewHolder.mTopicTxtFavorite.setText(StringUtils.getFormatNum(this.mTopicModelList.get(i).getCollectCount()));
        viewHolder.mTopicTxtRead.setText(StringUtils.getFormatNum(this.mTopicModelList.get(i).getViewCount()));
        if (this.mTopicModelList.get(i).getLikeed().intValue() == 1) {
            viewHolder.mTopicImgLike.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_like_pressed));
        } else {
            viewHolder.mTopicImgLike.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_like));
        }
        if (this.mTopicModelList.get(i).getCollected().intValue() == 1) {
            viewHolder.mTopicImgFavorite.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_collect_pressed));
        } else {
            viewHolder.mTopicImgFavorite.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_collect));
        }
        if (this.mTopicModelList.get(i).getViewed().intValue() == 1) {
            viewHolder.mTopicImgRead.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_read_pressed));
        } else {
            viewHolder.mTopicImgRead.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_read));
        }
        Glide.with((FragmentActivity) this.mContext).load(this.mTopicModelList.get(i).getUserHeadIcon()).placeholder(R.mipmap.ic_personal_head).into(viewHolder.mTopicImgHead);
        viewHolder.mTopicImgHead.setKing(this.mTopicModelList.get(i).getUserKing(), this.mContext, this.mTopicModelList.get(i).getGradeIcon());
        viewHolder.mTopicGridLayout.setAdapter(new CommAdapter<String>(this.mContext, this.mTopicModelList.get(i).getImages(), R.layout.item_topic_image) { // from class: com.hismith.hismithapp.adapter.DetailsTopicAdapter.1
            @Override // com.hismith.hismithapp.adapter.CommAdapter
            public void convert(XQuickViewHolder xQuickViewHolder, String str, final int i2) {
                Glide.with(this.mContext).load(str).placeholder(R.mipmap.hismith_image_loading).error(R.mipmap.hismith_image_load_err).into((ImageView) xQuickViewHolder.getView(R.id.item_topic_grid_image));
                xQuickViewHolder.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.DetailsTopicAdapter.1.1
                    @Override // android.view.View.OnClickListener
                    public void onClick(View view) {
                        ImagePreview.getInstance().setContext(AnonymousClass1.this.mContext).setIndex(i2).setImageList(((TopicModel) DetailsTopicAdapter.this.mTopicModelList.get(i)).getImages()).setFolderName("hismith/image").setEnableDragClose(true).start();
                    }
                });
            }
        });
        if (MyApplication.getApplication().getmUserModel() == null || MyApplication.getApplication().getmUserModel().getUserId() != this.mTopicModelList.get(i).getUserId().intValue()) {
            viewHolder.mTopicTxtDelete.setVisibility(8);
        } else {
            viewHolder.mTopicTxtDelete.setVisibility(0);
        }
        viewHolder.mTopicTxtDelete.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.DetailsTopicAdapter.2
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                DialogUtil.showConfirmDialog(DetailsTopicAdapter.this.mContext, "prompt", "deleteTopic", null, null, new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.DetailsTopicAdapter.2.1
                    @Override // android.view.View.OnClickListener
                    public void onClick(View view2) {
                        DetailsTopicAdapter.this.topicDelete(i);
                        DialogUtil.dismissDialog();
                    }
                });
            }
        });
        if (this.mTopicModelList.get(i).getLikeStatus() == 1) {
            viewHolder.mTopicLayoutLike.setEnabled(false);
            viewHolder.mTopicLayoutLike.setVisibility(4);
            viewHolder.mTopicImgProgressLike.setVisibility(0);
            AnimUtil.rotate(viewHolder.mTopicImgProgressLike);
        } else {
            viewHolder.mTopicLayoutLike.setEnabled(true);
            viewHolder.mTopicLayoutLike.setVisibility(0);
            viewHolder.mTopicImgProgressLike.clearAnimation();
            viewHolder.mTopicImgProgressLike.setVisibility(8);
        }
        if (this.mTopicModelList.get(i).getCollectStatus() == 1) {
            viewHolder.mTopicLayoutFavorite.setEnabled(false);
            viewHolder.mTopicLayoutFavorite.setVisibility(4);
            viewHolder.mTopicImgProgressFavorite.setVisibility(0);
            AnimUtil.rotate(viewHolder.mTopicImgProgressFavorite);
        } else {
            viewHolder.mTopicLayoutFavorite.setEnabled(true);
            viewHolder.mTopicLayoutFavorite.setVisibility(0);
            viewHolder.mTopicImgProgressFavorite.clearAnimation();
            viewHolder.mTopicImgProgressFavorite.setVisibility(8);
        }
        viewHolder.mTopicLayoutLike.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.DetailsTopicAdapter.3
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                viewHolder.mTopicLayoutLike.setEnabled(false);
                viewHolder.mTopicLayoutLike.setVisibility(4);
                viewHolder.mTopicImgProgressLike.setVisibility(0);
                AnimUtil.rotate(viewHolder.mTopicImgProgressLike);
                String str = ((TopicModel) DetailsTopicAdapter.this.mTopicModelList.get(i)).getLikeed().intValue() == 1 ? "unlike" : "like";
                ((TopicModel) DetailsTopicAdapter.this.mTopicModelList.get(i)).setLikeStatus(1);
                DetailsTopicAdapter detailsTopicAdapter = DetailsTopicAdapter.this;
                detailsTopicAdapter.topicLikeAction(i, ((TopicModel) detailsTopicAdapter.mTopicModelList.get(i)).getId(), str);
            }
        });
        viewHolder.mTopicLayoutFavorite.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.DetailsTopicAdapter.4
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                viewHolder.mTopicLayoutFavorite.setEnabled(false);
                viewHolder.mTopicLayoutFavorite.setVisibility(4);
                viewHolder.mTopicImgProgressFavorite.setVisibility(0);
                AnimUtil.rotate(viewHolder.mTopicImgProgressFavorite);
                String str = ((TopicModel) DetailsTopicAdapter.this.mTopicModelList.get(i)).getCollected().intValue() == 1 ? "uncollect" : "collect";
                ((TopicModel) DetailsTopicAdapter.this.mTopicModelList.get(i)).setCollectStatus(1);
                DetailsTopicAdapter detailsTopicAdapter = DetailsTopicAdapter.this;
                detailsTopicAdapter.topicCollectAction(i, ((TopicModel) detailsTopicAdapter.mTopicModelList.get(i)).getId(), str);
            }
        });
        viewHolder.mTopicLayoutItem.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.DetailsTopicAdapter.5
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (((TopicModel) DetailsTopicAdapter.this.mTopicModelList.get(i)).getId() != null) {
                    Intent intent = new Intent(DetailsTopicAdapter.this.mContext, (Class<?>) TopicContentActivity.class);
                    intent.putExtra("topicId", ((TopicModel) DetailsTopicAdapter.this.mTopicModelList.get(i)).getId());
                    DetailsTopicAdapter.this.mContext.startActivity(intent);
                }
            }
        });
    }

    /* renamed from: onBindViewHolder, reason: avoid collision after fix types in other method */
    public void onBindViewHolder2(ViewHolder viewHolder, int i, List<Object> list) {
        if (list.isEmpty()) {
            onBindViewHolder(viewHolder, i);
            return;
        }
        if ("like".equals(list.get(0))) {
            viewHolder.mTopicLayoutLike.setEnabled(true);
            viewHolder.mTopicImgProgressLike.clearAnimation();
            viewHolder.mTopicImgProgressLike.setVisibility(8);
            viewHolder.mTopicLayoutLike.setVisibility(0);
            viewHolder.mTopicTxtLike.setText(StringUtils.getFormatNum(this.mTopicModelList.get(i).getLikeCount()));
            if (this.mTopicModelList.get(i).getLikeed().intValue() == 1) {
                viewHolder.mTopicImgLike.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_like_pressed));
                return;
            } else {
                viewHolder.mTopicImgLike.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_like));
                return;
            }
        }
        if ("collect".equals(list.get(0))) {
            viewHolder.mTopicLayoutFavorite.setEnabled(true);
            viewHolder.mTopicImgProgressFavorite.clearAnimation();
            viewHolder.mTopicImgProgressFavorite.setVisibility(8);
            viewHolder.mTopicLayoutFavorite.setVisibility(0);
            viewHolder.mTopicTxtFavorite.setText(StringUtils.getFormatNum(this.mTopicModelList.get(i).getCollectCount()));
            if (this.mTopicModelList.get(i).getCollected().intValue() == 1) {
                viewHolder.mTopicImgFavorite.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_collect_pressed));
            } else {
                viewHolder.mTopicImgFavorite.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_collect));
            }
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_details_topic, viewGroup, false));
    }
}
