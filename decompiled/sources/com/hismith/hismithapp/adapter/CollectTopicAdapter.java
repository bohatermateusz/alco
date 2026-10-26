package com.hismith.hismithapp.adapter;

import android.content.Intent;
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
import com.hismith.hismithapp.PersonalDetailsActivity2;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.TopicContentActivity;
import com.hismith.hismithapp.UserDetailsActivity2;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.dialog.ReportSelectDialog;
import com.hismith.hismithapp.dialog.TopicActionDialog;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.TopicActionModel;
import com.hismith.hismithapp.model.TopicModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.DialogUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
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
public class CollectTopicAdapter extends RecyclerView.Adapter<ViewHolder> {
    private BaseActivity mContext;
    private List<TopicModel> mTopicModelList;
    private boolean isLike = false;
    private boolean isCollect = false;
    private RetrofitRequest baseRequest = new RetrofitRequest();

    /* renamed from: com.hismith.hismithapp.adapter.CollectTopicAdapter$3, reason: invalid class name */
    class AnonymousClass3 implements View.OnClickListener {
        final /* synthetic */ int val$position;

        AnonymousClass3(int i) {
            this.val$position = i;
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            if (((TopicModel) CollectTopicAdapter.this.mTopicModelList.get(this.val$position)).getUserId() != null) {
                TopicActionDialog topicActionDialog = new TopicActionDialog(CollectTopicAdapter.this.mContext);
                topicActionDialog.setOnClickItemActionListener(new TopicActionDialog.OnClickItemActionListener() { // from class: com.hismith.hismithapp.adapter.CollectTopicAdapter.3.1
                    @Override // com.hismith.hismithapp.dialog.TopicActionDialog.OnClickItemActionListener
                    public void btnClickDelete(final int i) {
                        DialogUtil.showConfirmDialog(CollectTopicAdapter.this.mContext, "prompt", "deleteTopic", null, null, new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.CollectTopicAdapter.3.1.3
                            @Override // android.view.View.OnClickListener
                            public void onClick(View view2) {
                                CollectTopicAdapter.this.topicDelete(i);
                                DialogUtil.dismissDialog();
                            }
                        });
                    }

                    @Override // com.hismith.hismithapp.dialog.TopicActionDialog.OnClickItemActionListener
                    public void btnClickNotInterested(final int i) {
                        DialogUtil.showConfirmDialog(CollectTopicAdapter.this.mContext, "prompt", "notInterestedTopic", null, "confirm", new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.CollectTopicAdapter.3.1.2
                            @Override // android.view.View.OnClickListener
                            public void onClick(View view2) {
                                CollectTopicAdapter.this.topicUninterestedAction(i);
                                DialogUtil.dismissDialog();
                            }
                        });
                    }

                    @Override // com.hismith.hismithapp.dialog.TopicActionDialog.OnClickItemActionListener
                    public void btnClickReport(final int i) {
                        new ReportSelectDialog(CollectTopicAdapter.this.mContext) { // from class: com.hismith.hismithapp.adapter.CollectTopicAdapter.3.1.1
                            @Override // com.hismith.hismithapp.dialog.ReportSelectDialog
                            public void btnSubmit(String str, String str2) {
                                CollectTopicAdapter.this.complaint(i, "topic", str, str2);
                            }
                        }.show();
                    }
                }, this.val$position);
                if (((TopicModel) CollectTopicAdapter.this.mTopicModelList.get(this.val$position)).getUserId().equals(Integer.valueOf(MyApplication.getApplication().getmUserModel().getUserId()))) {
                    topicActionDialog.showDelete().initDelete().show();
                } else {
                    topicActionDialog.hideDelete().initDelete().show();
                }
            }
        }
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private GridLayout mTopicGridLayout;
        private ImageView mTopicImgDown;
        private ImageView mTopicImgFavorite;
        private CircleImageView mTopicImgHead;
        private ImageView mTopicImgLike;
        private ImageView mTopicImgRead;
        private ImageView mTopicImgReview;
        private LinearLayout mTopicLayoutItem;
        private TextView mTopicTxtFavorite;
        private TextView mTopicTxtLike;
        private TextView mTopicTxtName;
        private TextView mTopicTxtRead;
        private TextView mTopicTxtReview;
        private TextView mTopicTxtTime;
        private TextView mTopicTxtTitle;

        public ViewHolder(View view) {
            super(view);
            this.mTopicLayoutItem = (LinearLayout) view.findViewById(R.id.topic_layout_item);
            this.mTopicImgHead = (CircleImageView) view.findViewById(R.id.topic_img_head);
            this.mTopicImgDown = (ImageView) view.findViewById(R.id.topic_img_down);
            this.mTopicImgLike = (ImageView) view.findViewById(R.id.topic_img_like);
            this.mTopicImgReview = (ImageView) view.findViewById(R.id.topic_img_review);
            this.mTopicImgFavorite = (ImageView) view.findViewById(R.id.topic_img_favorite);
            this.mTopicImgRead = (ImageView) view.findViewById(R.id.topic_img_read);
            this.mTopicTxtName = (TextView) view.findViewById(R.id.topic_txt_name);
            this.mTopicTxtTime = (TextView) view.findViewById(R.id.topic_txt_time);
            this.mTopicTxtTitle = (TextView) view.findViewById(R.id.topic_txt_title);
            this.mTopicTxtLike = (TextView) view.findViewById(R.id.topic_txt_like);
            this.mTopicTxtReview = (TextView) view.findViewById(R.id.topic_txt_review);
            this.mTopicTxtFavorite = (TextView) view.findViewById(R.id.topic_txt_favorite);
            this.mTopicTxtRead = (TextView) view.findViewById(R.id.topic_txt_read);
            this.mTopicGridLayout = (GridLayout) view.findViewById(R.id.topic_grid_layout);
        }
    }

    public CollectTopicAdapter(BaseActivity baseActivity, List<TopicModel> list) {
        this.mContext = baseActivity;
        this.mTopicModelList = list;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void complaint(int i, String str, String str2, String str3) {
        this.baseRequest.getApi().complaint((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), this.mTopicModelList.get(i).getId(), str, str2, str3).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this.mContext, null, false) { // from class: com.hismith.hismithapp.adapter.CollectTopicAdapter.9
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    ToastUtil.showShortToastCenter(CollectTopicAdapter.this.mContext, LanguageUtil.getLanguageByKey(CollectTopicAdapter.this.mContext, "reportedSuccess"));
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicCollectAction(final int i, Integer num, final String str) {
        this.isCollect = true;
        this.baseRequest.getApi().topicAction((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), num, str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicActionModel>(this.mContext, null, false) { // from class: com.hismith.hismithapp.adapter.CollectTopicAdapter.8
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                CollectTopicAdapter.this.isCollect = false;
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                CollectTopicAdapter.this.isCollect = false;
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicActionModel topicActionModel) {
                if (topicActionModel.getSuccess() == 1) {
                    String collectCount = topicActionModel.getCollectCount();
                    if ("collect".equals(str)) {
                        ((TopicModel) CollectTopicAdapter.this.mTopicModelList.get(i)).setCollected(1);
                    } else if ("uncollect".equals(str)) {
                        ((TopicModel) CollectTopicAdapter.this.mTopicModelList.get(i)).setCollected(0);
                    }
                    ((TopicModel) CollectTopicAdapter.this.mTopicModelList.get(i)).setCollectCount(collectCount);
                    CollectTopicAdapter.this.notifyItemChanged(i, "");
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicDelete(final int i) {
        Observable<HttpModel> observeOn = this.baseRequest.getApi().topicDel((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), this.mTopicModelList.get(i).getId()).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.mContext;
        observeOn.subscribe(new RxSubscribe<HttpModel>(baseActivity, baseActivity.progressDialog) { // from class: com.hismith.hismithapp.adapter.CollectTopicAdapter.11
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    CollectTopicAdapter.this.mTopicModelList.remove(i);
                    ToastUtil.showShortToastCenter(CollectTopicAdapter.this.mContext, LanguageUtil.getLanguageByKey(CollectTopicAdapter.this.mContext, "deletedMsg"));
                    CollectTopicAdapter.this.notifyItemRemoved(i);
                    CollectTopicAdapter collectTopicAdapter = CollectTopicAdapter.this;
                    collectTopicAdapter.notifyItemRangeChanged(i, collectTopicAdapter.mTopicModelList.size());
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicLikeAction(final int i, Integer num, final String str) {
        this.isLike = true;
        this.baseRequest.getApi().topicAction((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), num, str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicActionModel>(this.mContext, null, false) { // from class: com.hismith.hismithapp.adapter.CollectTopicAdapter.7
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                CollectTopicAdapter.this.isLike = false;
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                CollectTopicAdapter.this.isLike = false;
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicActionModel topicActionModel) {
                if (topicActionModel.getSuccess() == 1) {
                    String likeCount = topicActionModel.getLikeCount();
                    if ("like".equals(str)) {
                        ((TopicModel) CollectTopicAdapter.this.mTopicModelList.get(i)).setLikeed(1);
                    } else if ("unlike".equals(str)) {
                        ((TopicModel) CollectTopicAdapter.this.mTopicModelList.get(i)).setLikeed(0);
                    }
                    ((TopicModel) CollectTopicAdapter.this.mTopicModelList.get(i)).setLikeCount(likeCount);
                    CollectTopicAdapter.this.notifyItemChanged(i, "");
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicUninterestedAction(final int i) {
        this.baseRequest.getApi().topicAction((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), this.mTopicModelList.get(i).getId(), "uninterested").subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this.mContext, null, false) { // from class: com.hismith.hismithapp.adapter.CollectTopicAdapter.10
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    LogUtil.e("不感兴趣话题position:" + i + ", id:" + ((TopicModel) CollectTopicAdapter.this.mTopicModelList.get(i)).getId());
                    CollectTopicAdapter.this.mTopicModelList.remove(i);
                    CollectTopicAdapter.this.notifyItemRemoved(i);
                    CollectTopicAdapter collectTopicAdapter = CollectTopicAdapter.this;
                    collectTopicAdapter.notifyItemRangeChanged(i, collectTopicAdapter.mTopicModelList.size());
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
    public void onBindViewHolder(ViewHolder viewHolder, final int i) {
        viewHolder.mTopicTxtName.setText(this.mTopicModelList.get(i).getUserName());
        viewHolder.mTopicTxtTime.setText(TimeUtil.utc2Local(this.mTopicModelList.get(i).getCreateDate()));
        viewHolder.mTopicTxtTitle.setText(this.mTopicModelList.get(i).getTitle());
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
        viewHolder.mTopicGridLayout.setAdapter(new CommAdapter<String>(this.mContext, this.mTopicModelList.get(i).getImages(), R.layout.item_topic_image) { // from class: com.hismith.hismithapp.adapter.CollectTopicAdapter.1
            @Override // com.hismith.hismithapp.adapter.CommAdapter
            public void convert(XQuickViewHolder xQuickViewHolder, String str, int i2) {
                Glide.with(this.mContext).load(str).into((ImageView) xQuickViewHolder.getView(R.id.item_topic_grid_image));
                xQuickViewHolder.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.CollectTopicAdapter.1.1
                    @Override // android.view.View.OnClickListener
                    public void onClick(View view) {
                    }
                });
            }
        });
        viewHolder.mTopicImgHead.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.CollectTopicAdapter.2
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (((TopicModel) CollectTopicAdapter.this.mTopicModelList.get(i)).getUserId() != null) {
                    if (MyApplication.getApplication().getmUserModel() != null && ((TopicModel) CollectTopicAdapter.this.mTopicModelList.get(i)).getUserId().intValue() == MyApplication.getApplication().getmUserModel().getUserId()) {
                        CollectTopicAdapter.this.mContext.startActivity(new Intent(CollectTopicAdapter.this.mContext, (Class<?>) PersonalDetailsActivity2.class));
                    } else {
                        Intent intent = new Intent(CollectTopicAdapter.this.mContext, (Class<?>) UserDetailsActivity2.class);
                        intent.putExtra("userId", ((TopicModel) CollectTopicAdapter.this.mTopicModelList.get(i)).getUserId());
                        CollectTopicAdapter.this.mContext.startActivity(intent);
                    }
                }
            }
        });
        viewHolder.mTopicImgDown.setOnClickListener(new AnonymousClass3(i));
        viewHolder.mTopicImgLike.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.CollectTopicAdapter.4
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (CollectTopicAdapter.this.isLike) {
                    return;
                }
                String str = ((TopicModel) CollectTopicAdapter.this.mTopicModelList.get(i)).getLikeed().intValue() == 1 ? "unlike" : "like";
                CollectTopicAdapter collectTopicAdapter = CollectTopicAdapter.this;
                collectTopicAdapter.topicLikeAction(i, ((TopicModel) collectTopicAdapter.mTopicModelList.get(i)).getId(), str);
            }
        });
        viewHolder.mTopicImgFavorite.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.CollectTopicAdapter.5
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (CollectTopicAdapter.this.isCollect) {
                    return;
                }
                String str = ((TopicModel) CollectTopicAdapter.this.mTopicModelList.get(i)).getCollected().intValue() == 1 ? "uncollect" : "collect";
                CollectTopicAdapter collectTopicAdapter = CollectTopicAdapter.this;
                collectTopicAdapter.topicCollectAction(i, ((TopicModel) collectTopicAdapter.mTopicModelList.get(i)).getId(), str);
            }
        });
        viewHolder.mTopicLayoutItem.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.CollectTopicAdapter.6
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (((TopicModel) CollectTopicAdapter.this.mTopicModelList.get(i)).getId() != null) {
                    Intent intent = new Intent(CollectTopicAdapter.this.mContext, (Class<?>) TopicContentActivity.class);
                    intent.putExtra("topicId", ((TopicModel) CollectTopicAdapter.this.mTopicModelList.get(i)).getId());
                    CollectTopicAdapter.this.mContext.startActivity(intent);
                }
            }
        });
    }

    /* renamed from: onBindViewHolder, reason: avoid collision after fix types in other method */
    public void onBindViewHolder2(ViewHolder viewHolder, int i, List<Object> list) {
        super.onBindViewHolder((CollectTopicAdapter) viewHolder, i, list);
        if (list.isEmpty()) {
            onBindViewHolder(viewHolder, i);
            return;
        }
        viewHolder.mTopicTxtLike.setText(StringUtils.getFormatNum(this.mTopicModelList.get(i).getLikeCount()));
        if (this.mTopicModelList.get(i).getLikeed().intValue() == 1) {
            viewHolder.mTopicImgLike.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_like_pressed));
        } else {
            viewHolder.mTopicImgLike.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_like));
        }
        viewHolder.mTopicTxtFavorite.setText(StringUtils.getFormatNum(this.mTopicModelList.get(i).getCollectCount()));
        if (this.mTopicModelList.get(i).getCollected().intValue() == 1) {
            viewHolder.mTopicImgFavorite.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_collect_pressed));
        } else {
            viewHolder.mTopicImgFavorite.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_collect));
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_topic, viewGroup, false));
    }
}
