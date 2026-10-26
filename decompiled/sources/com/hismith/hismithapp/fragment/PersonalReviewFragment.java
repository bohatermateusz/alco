package com.hismith.hismithapp.fragment;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.MyReviewAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.dialog.ReportSelectDialog;
import com.hismith.hismithapp.dialog.TopicActionDialog;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.TopicActionModel;
import com.hismith.hismithapp.model.TopicListModel;
import com.hismith.hismithapp.model.TopicModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.DialogUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnRefreshLoadMoreListener;
import io.reactivex.Observable;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class PersonalReviewFragment extends LazyLoadFragment {
    private BaseActivity baseActivity;
    private RetrofitRequest baseRequest;
    private MyReviewAdapter mAdapter;
    private LinearLayoutManager mLinearLayoutManager;
    private RecyclerView mRecyclerView;
    private SmartRefreshLayout mRefreshLayout;
    private List<TopicModel> mTopicModelList;
    private int page = 1;
    private int totalPage = 1;

    static /* synthetic */ int access$008(PersonalReviewFragment personalReviewFragment) {
        int i = personalReviewFragment.page;
        personalReviewFragment.page = i + 1;
        return i;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void complaint(int i, String str, String str2, String str3) {
        Observable<HttpModel> observeOn = this.baseRequest.getApi().complaint((String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token), this.mTopicModelList.get(i).getId(), str, str2, str3).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.baseActivity;
        observeOn.subscribe(new RxSubscribe<HttpModel>(baseActivity, baseActivity.progressDialog, true) { // from class: com.hismith.hismithapp.fragment.PersonalReviewFragment.7
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    ToastUtil.showShortToastCenter(PersonalReviewFragment.this.baseActivity, LanguageUtil.getLanguageByKey(PersonalReviewFragment.this.baseActivity, "reportedSuccess"));
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void getUserCollectTopic(int i) {
        this.baseRequest.getApi().myTopic((String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token), "review", Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicListModel>(this.baseActivity, null, false) { // from class: com.hismith.hismithapp.fragment.PersonalReviewFragment.6
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                PersonalReviewFragment.this.mRefreshLayout.finishRefresh();
                PersonalReviewFragment.this.mRefreshLayout.finishLoadMore();
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                PersonalReviewFragment.this.mRefreshLayout.finishRefresh();
                PersonalReviewFragment.this.mRefreshLayout.finishLoadMore();
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicListModel topicListModel) {
                if (topicListModel.getSuccess() == 1) {
                    if (topicListModel.getPage().intValue() == 1) {
                        PersonalReviewFragment.this.mTopicModelList.clear();
                    }
                    PersonalReviewFragment.this.totalPage = topicListModel.getTotalPage().intValue();
                    if (topicListModel.getTopics() != null) {
                        PersonalReviewFragment.this.mTopicModelList.addAll(Arrays.asList(topicListModel.getTopics()));
                        PersonalReviewFragment.this.mAdapter.notifyDataSetChanged();
                    }
                }
            }
        });
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this.baseActivity);
        this.mLinearLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mTopicModelList = new ArrayList();
        MyReviewAdapter myReviewAdapter = new MyReviewAdapter(this.baseActivity, this.mTopicModelList);
        this.mAdapter = myReviewAdapter;
        this.mRecyclerView.setAdapter(myReviewAdapter);
        this.mRefreshLayout.autoRefresh();
    }

    private void initView(View view) {
        this.mRefreshLayout = (SmartRefreshLayout) view.findViewById(R.id.personal_topic_refreshLayout);
        this.mRecyclerView = (RecyclerView) view.findViewById(R.id.personal_topic_recycler_view);
    }

    private void setListener() {
        this.mRefreshLayout.setOnRefreshLoadMoreListener(new OnRefreshLoadMoreListener() { // from class: com.hismith.hismithapp.fragment.PersonalReviewFragment.1
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout) {
                int i = PersonalReviewFragment.this.page;
                int i2 = PersonalReviewFragment.this.totalPage;
                PersonalReviewFragment personalReviewFragment = PersonalReviewFragment.this;
                if (i >= i2) {
                    personalReviewFragment.mRefreshLayout.finishLoadMore(1000);
                    return;
                }
                PersonalReviewFragment.access$008(personalReviewFragment);
                PersonalReviewFragment personalReviewFragment2 = PersonalReviewFragment.this;
                personalReviewFragment2.getUserCollectTopic(personalReviewFragment2.page);
            }

            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout) {
                PersonalReviewFragment.this.page = 1;
                PersonalReviewFragment personalReviewFragment = PersonalReviewFragment.this;
                personalReviewFragment.getUserCollectTopic(personalReviewFragment.page);
            }
        });
        this.mAdapter.setOnDownClickListener(new MyReviewAdapter.OnDownClickListener() { // from class: com.hismith.hismithapp.fragment.PersonalReviewFragment.2
            @Override // com.hismith.hismithapp.adapter.MyReviewAdapter.OnDownClickListener
            public void onDownClick(int i) {
                PersonalReviewFragment.this.showDownDialog(i);
            }
        });
        this.mAdapter.setOnLikeClickListener(new MyReviewAdapter.OnLikeClickListener() { // from class: com.hismith.hismithapp.fragment.PersonalReviewFragment.3
            @Override // com.hismith.hismithapp.adapter.MyReviewAdapter.OnLikeClickListener
            public void onLikeClick(int i) {
                String str;
                if (((TopicModel) PersonalReviewFragment.this.mTopicModelList.get(i)).getLikeed().intValue() == 1) {
                    LogUtil.e("触发取消点赞");
                    str = "unlike";
                } else {
                    LogUtil.e("触发点赞");
                    str = "like";
                }
                PersonalReviewFragment personalReviewFragment = PersonalReviewFragment.this;
                personalReviewFragment.topicLikeAction(i, ((TopicModel) personalReviewFragment.mTopicModelList.get(i)).getId(), str);
            }
        });
        this.mAdapter.setOnCollectClickListener(new MyReviewAdapter.OnCollectClickListener() { // from class: com.hismith.hismithapp.fragment.PersonalReviewFragment.4
            @Override // com.hismith.hismithapp.adapter.MyReviewAdapter.OnCollectClickListener
            public void onCollectClick(int i) {
                String str;
                if (((TopicModel) PersonalReviewFragment.this.mTopicModelList.get(i)).getCollected().intValue() == 1) {
                    LogUtil.e("触发取消收藏");
                    str = "uncollect";
                } else {
                    LogUtil.e("触发收藏");
                    str = "collect";
                }
                PersonalReviewFragment personalReviewFragment = PersonalReviewFragment.this;
                personalReviewFragment.topicCollectAction(i, ((TopicModel) personalReviewFragment.mTopicModelList.get(i)).getId(), str);
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void showDownDialog(int i) {
        if (this.mTopicModelList.get(i).getUserId() != null) {
            TopicActionDialog topicActionDialog = new TopicActionDialog(this.baseActivity);
            topicActionDialog.setOnClickItemActionListener(new TopicActionDialog.OnClickItemActionListener() { // from class: com.hismith.hismithapp.fragment.PersonalReviewFragment.5
                @Override // com.hismith.hismithapp.dialog.TopicActionDialog.OnClickItemActionListener
                public void btnClickDelete(final int i2) {
                    DialogUtil.showConfirmDialog(PersonalReviewFragment.this.baseActivity, "prompt", "deleteTopic", null, null, new View.OnClickListener() { // from class: com.hismith.hismithapp.fragment.PersonalReviewFragment.5.3
                        @Override // android.view.View.OnClickListener
                        public void onClick(View view) {
                            PersonalReviewFragment.this.topicDelete(i2);
                            DialogUtil.dismissDialog();
                        }
                    });
                }

                @Override // com.hismith.hismithapp.dialog.TopicActionDialog.OnClickItemActionListener
                public void btnClickNotInterested(final int i2) {
                    DialogUtil.showConfirmDialog(PersonalReviewFragment.this.baseActivity, "prompt", "notInterestedTopic", null, "confirm", new View.OnClickListener() { // from class: com.hismith.hismithapp.fragment.PersonalReviewFragment.5.2
                        @Override // android.view.View.OnClickListener
                        public void onClick(View view) {
                            PersonalReviewFragment.this.topicUninterestedAction(i2);
                            DialogUtil.dismissDialog();
                        }
                    });
                }

                @Override // com.hismith.hismithapp.dialog.TopicActionDialog.OnClickItemActionListener
                public void btnClickReport(final int i2) {
                    new ReportSelectDialog(PersonalReviewFragment.this.baseActivity) { // from class: com.hismith.hismithapp.fragment.PersonalReviewFragment.5.1
                        @Override // com.hismith.hismithapp.dialog.ReportSelectDialog
                        public void btnSubmit(String str, String str2) {
                            PersonalReviewFragment.this.complaint(i2, "topic", str, str2);
                        }
                    }.showDialog();
                }
            }, i);
            if (this.mTopicModelList.get(i).getUserId().equals(Integer.valueOf(MyApplication.getApplication().getmUserModel().getUserId()))) {
                topicActionDialog.showDelete().initDelete().show();
            } else {
                topicActionDialog.hideDelete().initDelete().show();
            }
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicCollectAction(final int i, Integer num, final String str) {
        this.baseRequest.getApi().topicAction((String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token), num, str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicActionModel>(this.baseActivity, null, false) { // from class: com.hismith.hismithapp.fragment.PersonalReviewFragment.11
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                ((TopicModel) PersonalReviewFragment.this.mTopicModelList.get(i)).setCollectStatus(0);
                PersonalReviewFragment.this.mAdapter.notifyItemChanged(i, "collect");
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                ((TopicModel) PersonalReviewFragment.this.mTopicModelList.get(i)).setCollectStatus(0);
                PersonalReviewFragment.this.mAdapter.notifyItemChanged(i, "collect");
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicActionModel topicActionModel) {
                if (topicActionModel.getSuccess() == 1) {
                    String collectCount = topicActionModel.getCollectCount();
                    if ("collect".equals(str)) {
                        ((TopicModel) PersonalReviewFragment.this.mTopicModelList.get(i)).setCollected(1);
                    } else if ("uncollect".equals(str)) {
                        ((TopicModel) PersonalReviewFragment.this.mTopicModelList.get(i)).setCollected(0);
                    }
                    ((TopicModel) PersonalReviewFragment.this.mTopicModelList.get(i)).setCollectCount(collectCount);
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicDelete(final int i) {
        Observable<HttpModel> observeOn = this.baseRequest.getApi().topicDel((String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token), this.mTopicModelList.get(i).getId()).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.baseActivity;
        observeOn.subscribe(new RxSubscribe<HttpModel>(baseActivity, baseActivity.progressDialog) { // from class: com.hismith.hismithapp.fragment.PersonalReviewFragment.9
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    PersonalReviewFragment.this.mTopicModelList.remove(i);
                    ToastUtil.showShortToastCenter(PersonalReviewFragment.this.baseActivity, LanguageUtil.getLanguageByKey(PersonalReviewFragment.this.baseActivity, "deletedMsg"));
                    PersonalReviewFragment.this.mAdapter.notifyItemRemoved(i);
                    PersonalReviewFragment.this.mAdapter.notifyItemRangeChanged(i, PersonalReviewFragment.this.mTopicModelList.size());
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicLikeAction(final int i, Integer num, final String str) {
        this.baseRequest.getApi().topicAction((String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token), num, str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicActionModel>(this.baseActivity, null, false) { // from class: com.hismith.hismithapp.fragment.PersonalReviewFragment.10
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                ((TopicModel) PersonalReviewFragment.this.mTopicModelList.get(i)).setLikeStatus(0);
                PersonalReviewFragment.this.mAdapter.notifyItemChanged(i, "like");
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                ((TopicModel) PersonalReviewFragment.this.mTopicModelList.get(i)).setLikeStatus(0);
                PersonalReviewFragment.this.mAdapter.notifyItemChanged(i, "like");
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicActionModel topicActionModel) {
                if (topicActionModel.getSuccess() == 1) {
                    String likeCount = topicActionModel.getLikeCount();
                    if ("like".equals(str)) {
                        ((TopicModel) PersonalReviewFragment.this.mTopicModelList.get(i)).setLikeed(1);
                    } else if ("unlike".equals(str)) {
                        ((TopicModel) PersonalReviewFragment.this.mTopicModelList.get(i)).setLikeed(0);
                    }
                    ((TopicModel) PersonalReviewFragment.this.mTopicModelList.get(i)).setLikeCount(likeCount);
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicUninterestedAction(final int i) {
        this.baseRequest.getApi().topicAction((String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token), this.mTopicModelList.get(i).getId(), "uninterested").subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this.baseActivity, null, false) { // from class: com.hismith.hismithapp.fragment.PersonalReviewFragment.8
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    PersonalReviewFragment.this.mTopicModelList.remove(i);
                    PersonalReviewFragment.this.mAdapter.notifyItemRemoved(i);
                    PersonalReviewFragment.this.mAdapter.notifyItemRangeChanged(i, PersonalReviewFragment.this.mTopicModelList.size());
                }
            }
        });
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment
    public void loadData() {
        SmartRefreshLayout smartRefreshLayout = this.mRefreshLayout;
        if (smartRefreshLayout != null) {
            smartRefreshLayout.autoRefresh();
        }
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.baseActivity = (BaseActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_personal_review, (ViewGroup) null);
        initView(inflate);
        return inflate;
    }
}
