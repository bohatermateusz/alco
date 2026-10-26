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
public class PersonalCollectFragment extends LazyLoadFragment {
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
    public Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.PersonalCollectFragment.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            if (message.what == 1 && PersonalCollectFragment.this.mRecyclerView != null) {
                PersonalCollectFragment.this.mRecyclerView.scrollBy(0, 1);
                PersonalCollectFragment.this.mRecyclerView.scrollBy(0, -1);
            }
            return false;
        }
    });

    static /* synthetic */ int access$108(PersonalCollectFragment personalCollectFragment) {
        int i = personalCollectFragment.page;
        personalCollectFragment.page = i + 1;
        return i;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void getUserCollectTopic(int i) {
        this.baseRequest.getApi().myTopic((String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token), "collect", Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicListModel>(this.baseActivity, null, false) { // from class: com.hismith.hismithapp.fragment.PersonalCollectFragment.3
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                PersonalCollectFragment.this.mRefreshLayout.finishRefresh();
                PersonalCollectFragment.this.mRefreshLayout.finishLoadMore();
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                PersonalCollectFragment.this.mRefreshLayout.finishRefresh();
                PersonalCollectFragment.this.mRefreshLayout.finishLoadMore();
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicListModel topicListModel) {
                if (topicListModel.getSuccess() == 1) {
                    if (topicListModel.getPage().intValue() == 1) {
                        PersonalCollectFragment.this.mTopicModelList.clear();
                    }
                    PersonalCollectFragment.this.totalPage = topicListModel.getTotalPage().intValue();
                    if (topicListModel.getTopics() != null) {
                        PersonalCollectFragment.this.mTopicModelList.addAll(Arrays.asList(topicListModel.getTopics()));
                        PersonalCollectFragment.this.mAdapter.notifyDataSetChanged();
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
        this.mRefreshLayout = (SmartRefreshLayout) view.findViewById(R.id.personal_collect_refreshLayout);
        this.mRecyclerView = (RecyclerView) view.findViewById(R.id.personal_collect_recycler_view);
    }

    private void setListener() {
        this.mRefreshLayout.setOnRefreshLoadMoreListener(new OnRefreshLoadMoreListener() { // from class: com.hismith.hismithapp.fragment.PersonalCollectFragment.2
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout) {
                int i = PersonalCollectFragment.this.page;
                int i2 = PersonalCollectFragment.this.totalPage;
                PersonalCollectFragment personalCollectFragment = PersonalCollectFragment.this;
                if (i >= i2) {
                    personalCollectFragment.mRefreshLayout.finishLoadMore(1000);
                    return;
                }
                PersonalCollectFragment.access$108(personalCollectFragment);
                PersonalCollectFragment personalCollectFragment2 = PersonalCollectFragment.this;
                personalCollectFragment2.getUserCollectTopic(personalCollectFragment2.page);
            }

            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout) {
                PersonalCollectFragment.this.page = 1;
                PersonalCollectFragment personalCollectFragment = PersonalCollectFragment.this;
                personalCollectFragment.getUserCollectTopic(personalCollectFragment.page);
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
        View inflate = layoutInflater.inflate(R.layout.fragment_personal_collect, (ViewGroup) null);
        initView(inflate);
        return inflate;
    }
}
