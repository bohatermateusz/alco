package com.hismith.hismithapp.fragment;

import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.CommunityAdapter;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.TopicListModel;
import com.hismith.hismithapp.model.TopicModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnRefreshLoadMoreListener;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class PersonalTopicFragment extends LazyLoadFragment {
    public static final int WHAT_NOTIFYDATA = 1;
    private BaseActivity baseActivity;
    private RetrofitRequest baseRequest;
    private CommunityAdapter mAdapter;
    private LinearLayoutManager mLinearLayoutManager;
    private RecyclerView mRecyclerView;
    private SmartRefreshLayout mRefreshLayout;
    private List<TopicModel> mTopicModelList;
    private int page = 1;
    private int totalPage = 1;
    public Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.PersonalTopicFragment.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            if (message.what == 1 && PersonalTopicFragment.this.mRecyclerView != null) {
                PersonalTopicFragment.this.mRecyclerView.scrollBy(0, 1);
                PersonalTopicFragment.this.mRecyclerView.scrollBy(0, -1);
            }
            return false;
        }
    });

    static /* synthetic */ int access$108(PersonalTopicFragment personalTopicFragment) {
        int i = personalTopicFragment.page;
        personalTopicFragment.page = i + 1;
        return i;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void getUserCollectTopic(int i) {
        this.baseRequest.getApi().myTopic((String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token), "my", Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicListModel>(this.baseActivity, null, false) { // from class: com.hismith.hismithapp.fragment.PersonalTopicFragment.3
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                PersonalTopicFragment.this.mRefreshLayout.finishRefresh();
                PersonalTopicFragment.this.mRefreshLayout.finishLoadMore();
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                PersonalTopicFragment.this.mRefreshLayout.finishRefresh();
                PersonalTopicFragment.this.mRefreshLayout.finishLoadMore();
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicListModel topicListModel) {
                if (topicListModel.getSuccess() == 1) {
                    if (topicListModel.getPage().intValue() == 1) {
                        PersonalTopicFragment.this.mTopicModelList.clear();
                    }
                    PersonalTopicFragment.this.totalPage = topicListModel.getTotalPage().intValue();
                    if (topicListModel.getTopics() != null) {
                        PersonalTopicFragment.this.mTopicModelList.addAll(Arrays.asList(topicListModel.getTopics()));
                        PersonalTopicFragment.this.mAdapter.notifyDataSetChanged();
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
        CommunityAdapter communityAdapter = new CommunityAdapter(this.baseActivity, this.mTopicModelList);
        this.mAdapter = communityAdapter;
        this.mRecyclerView.setAdapter(communityAdapter);
        this.mRefreshLayout.autoRefresh();
    }

    private void initView(View view) {
        this.mRefreshLayout = (SmartRefreshLayout) view.findViewById(R.id.personal_topic_refreshLayout);
        this.mRecyclerView = (RecyclerView) view.findViewById(R.id.personal_topic_recycler_view);
    }

    private void setListener() {
        this.mRefreshLayout.setOnRefreshLoadMoreListener(new OnRefreshLoadMoreListener() { // from class: com.hismith.hismithapp.fragment.PersonalTopicFragment.2
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout) {
                int i = PersonalTopicFragment.this.page;
                int i2 = PersonalTopicFragment.this.totalPage;
                PersonalTopicFragment personalTopicFragment = PersonalTopicFragment.this;
                if (i >= i2) {
                    personalTopicFragment.mRefreshLayout.finishLoadMore(1000);
                    return;
                }
                PersonalTopicFragment.access$108(personalTopicFragment);
                PersonalTopicFragment personalTopicFragment2 = PersonalTopicFragment.this;
                personalTopicFragment2.getUserCollectTopic(personalTopicFragment2.page);
            }

            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout) {
                PersonalTopicFragment.this.page = 1;
                PersonalTopicFragment personalTopicFragment = PersonalTopicFragment.this;
                personalTopicFragment.getUserCollectTopic(personalTopicFragment.page);
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
        View inflate = layoutInflater.inflate(R.layout.fragment_personal_topic, (ViewGroup) null);
        initView(inflate);
        return inflate;
    }
}
