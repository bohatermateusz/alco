package com.hismith.hismithapp.fragment;

import android.content.Intent;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.PersonalDetailsActivity2;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.TopicContentActivity;
import com.hismith.hismithapp.UserDetailsActivity2;
import com.hismith.hismithapp.adapter.CommunityAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.dialog.NetErrorDialog;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.TopicListModel;
import com.hismith.hismithapp.model.TopicModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnLoadMoreListener;
import com.scwang.smart.refresh.layout.listener.OnRefreshListener;
import io.reactivex.Observable;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class CommunityTopicFragment extends LazyLoadFragment {
    private BaseActivity baseActivity;
    private RetrofitRequest baseRequest;
    private CommunityAdapter mAdapter;
    private ActivityResultLauncher<Intent> mBlockLauncher;
    private FrameLayout mCommunityLayout;
    private LinearLayoutManager mLayoutManager;
    private RecyclerView mRecyclerView;
    private RefreshLayout mRefreshLayout;
    private String mSearchKey;
    private String menuId;
    private List<TopicModel> mTopicList = new ArrayList();
    private String requestId = "";
    private boolean isLazyLoad = false;
    private int viewSex = 0;

    public static CommunityTopicFragment getInstance(String str) {
        CommunityTopicFragment communityTopicFragment = new CommunityTopicFragment();
        Bundle bundle = new Bundle();
        bundle.putString("menuId", str);
        communityTopicFragment.setArguments(bundle);
        return communityTopicFragment;
    }

    public static CommunityTopicFragment getInstance(String str, String str2) {
        CommunityTopicFragment communityTopicFragment = new CommunityTopicFragment();
        Bundle bundle = new Bundle();
        bundle.putString("menuId", str);
        bundle.putString("mSearchKey", str2);
        communityTopicFragment.setArguments(bundle);
        return communityTopicFragment;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void getLoadMoreTopicList() {
        Observable<TopicListModel> observeOn = this.baseRequest.getApi().topicList((String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token), this.mSearchKey, this.menuId, 0, this.requestId).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.baseActivity;
        observeOn.subscribe(new RxSubscribe<TopicListModel>(baseActivity, baseActivity.progressDialog, false) { // from class: com.hismith.hismithapp.fragment.CommunityTopicFragment.7
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                CommunityTopicFragment.this.mRefreshLayout.finishLoadMore();
                NetErrorDialog.dissErrorDialog(CommunityTopicFragment.this.baseActivity, CommunityTopicFragment.this.mCommunityLayout);
                if (CommunityTopicFragment.this.mRecyclerView.getVisibility() != 0) {
                    CommunityTopicFragment.this.mRecyclerView.setVisibility(0);
                }
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                CommunityTopicFragment.this.mRefreshLayout.finishLoadMore();
                CommunityTopicFragment.this.mRecyclerView.setVisibility(8);
                NetErrorDialog.showErrorDialog(CommunityTopicFragment.this.baseActivity, CommunityTopicFragment.this.mCommunityLayout, new NetErrorDialog.OnResetClickListener() { // from class: com.hismith.hismithapp.fragment.CommunityTopicFragment.7.1
                    @Override // com.hismith.hismithapp.dialog.NetErrorDialog.OnResetClickListener
                    public void onResetClick() {
                        CommunityTopicFragment.this.mRefreshLayout.autoLoadMore();
                    }
                });
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicListModel topicListModel) {
                if (topicListModel.getSuccess() == 1) {
                    CommunityTopicFragment.this.requestId = topicListModel.getRequestId();
                    if (topicListModel.getTopics() != null && topicListModel.getTopics().length > 0) {
                        CommunityTopicFragment.this.mTopicList.addAll(Arrays.asList(topicListModel.getTopics()));
                    }
                    if (CommunityTopicFragment.this.mAdapter != null) {
                        CommunityTopicFragment.this.mAdapter.setShowEmptyView(true);
                        CommunityTopicFragment.this.mAdapter.notifyDataSetChanged();
                    }
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void getTopicList() {
        Observable<TopicListModel> observeOn = this.baseRequest.getApi().topicList((String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token), this.mSearchKey, this.menuId, 1, "").subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.baseActivity;
        observeOn.subscribe(new RxSubscribe<TopicListModel>(baseActivity, baseActivity.progressDialog, false) { // from class: com.hismith.hismithapp.fragment.CommunityTopicFragment.6
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                CommunityTopicFragment.this.mRefreshLayout.finishRefresh();
                NetErrorDialog.dissErrorDialog(CommunityTopicFragment.this.baseActivity, CommunityTopicFragment.this.mCommunityLayout);
                if (CommunityTopicFragment.this.mRecyclerView.getVisibility() != 0) {
                    CommunityTopicFragment.this.mRecyclerView.setVisibility(0);
                }
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                CommunityTopicFragment.this.mRefreshLayout.finishRefresh();
                CommunityTopicFragment.this.mRecyclerView.setVisibility(8);
                NetErrorDialog.showErrorDialog(CommunityTopicFragment.this.baseActivity, CommunityTopicFragment.this.mCommunityLayout, new NetErrorDialog.OnResetClickListener() { // from class: com.hismith.hismithapp.fragment.CommunityTopicFragment.6.1
                    @Override // com.hismith.hismithapp.dialog.NetErrorDialog.OnResetClickListener
                    public void onResetClick() {
                        CommunityTopicFragment.this.mRefreshLayout.autoRefresh();
                    }
                });
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicListModel topicListModel) {
                if (topicListModel.getSuccess() == 1) {
                    CommunityTopicFragment.this.requestId = topicListModel.getRequestId();
                    CommunityTopicFragment.this.mTopicList.clear();
                    if (topicListModel.getTopics() != null && topicListModel.getTopics().length > 0) {
                        CommunityTopicFragment.this.mTopicList.addAll(Arrays.asList(topicListModel.getTopics()));
                    }
                    if (CommunityTopicFragment.this.mAdapter != null) {
                        CommunityTopicFragment.this.mAdapter.setShowEmptyView(true);
                        CommunityTopicFragment.this.mAdapter.notifyDataSetChanged();
                    }
                }
            }
        });
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this.baseActivity);
        this.mLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        CommunityAdapter communityAdapter = new CommunityAdapter(this.baseActivity, this.mTopicList);
        this.mAdapter = communityAdapter;
        this.mRecyclerView.setAdapter(communityAdapter);
        if (MyApplication.getApplication().getmUserModel() != null) {
            this.viewSex = MyApplication.getApplication().getmUserModel().getViewSex();
        }
        this.mRefreshLayout.autoRefresh();
    }

    private void initActivityResult() {
        this.mBlockLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), new ActivityResultCallback<ActivityResult>() { // from class: com.hismith.hismithapp.fragment.CommunityTopicFragment.5
            @Override // androidx.activity.result.ActivityResultCallback
            public void onActivityResult(ActivityResult activityResult) {
                LogUtil.e("onActivityResult:" + activityResult.getResultCode());
                if (activityResult.getResultCode() != 108 || CommunityTopicFragment.this.mRefreshLayout == null) {
                    return;
                }
                CommunityTopicFragment.this.mRefreshLayout.autoRefresh();
            }
        });
    }

    private void setListener() {
        this.mAdapter.setOnItemClickListener(new CommunityAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.fragment.CommunityTopicFragment.1
            @Override // com.hismith.hismithapp.adapter.CommunityAdapter.OnItemClickListener
            public void onItemClick(int i) {
                if (((TopicModel) CommunityTopicFragment.this.mTopicList.get(i)).getId() != null) {
                    Intent intent = new Intent(CommunityTopicFragment.this.baseActivity, (Class<?>) TopicContentActivity.class);
                    intent.putExtra("topicId", ((TopicModel) CommunityTopicFragment.this.mTopicList.get(i)).getId());
                    CommunityTopicFragment.this.mBlockLauncher.launch(intent);
                }
            }
        });
        this.mAdapter.setOnUserInfoListener(new CommunityAdapter.OnUserInfoListener() { // from class: com.hismith.hismithapp.fragment.CommunityTopicFragment.2
            @Override // com.hismith.hismithapp.adapter.CommunityAdapter.OnUserInfoListener
            public void onUserInfoClick(int i) {
                if (MyApplication.getApplication().getmUserModel() != null) {
                    if (i == MyApplication.getApplication().getmUserModel().getUserId()) {
                        CommunityTopicFragment.this.startActivity(new Intent(CommunityTopicFragment.this.baseActivity, (Class<?>) PersonalDetailsActivity2.class));
                    } else {
                        Intent intent = new Intent(CommunityTopicFragment.this.baseActivity, (Class<?>) UserDetailsActivity2.class);
                        intent.putExtra("userId", i);
                        CommunityTopicFragment.this.mBlockLauncher.launch(intent);
                    }
                }
            }
        });
        this.mRefreshLayout.setOnRefreshListener(new OnRefreshListener() { // from class: com.hismith.hismithapp.fragment.CommunityTopicFragment.3
            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout) {
                CommunityTopicFragment.this.getTopicList();
            }
        });
        this.mRefreshLayout.setOnLoadMoreListener(new OnLoadMoreListener() { // from class: com.hismith.hismithapp.fragment.CommunityTopicFragment.4
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout) {
                CommunityTopicFragment.this.getLoadMoreTopicList();
            }
        });
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        this.isLazyLoad = true;
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.baseActivity = (BaseActivity) getActivity();
        initActivityResult();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_community_topic, (ViewGroup) null);
        this.mCommunityLayout = (FrameLayout) inflate.findViewById(R.id.community_layout);
        this.mRefreshLayout = (RefreshLayout) inflate.findViewById(R.id.community_refreshLayout);
        this.mRecyclerView = (RecyclerView) inflate.findViewById(R.id.community_recycler);
        Bundle arguments = getArguments();
        if (arguments != null) {
            this.menuId = arguments.getString("menuId");
            this.mSearchKey = arguments.getString("mSearchKey");
        }
        return inflate;
    }

    @Override // androidx.fragment.app.Fragment
    public void onResume() {
        super.onResume();
        if (!this.isLazyLoad || MyApplication.getApplication().getmUserModel() == null || MyApplication.getApplication().getmUserModel().getViewSex() == this.viewSex) {
            return;
        }
        this.viewSex = MyApplication.getApplication().getmUserModel().getViewSex();
        RefreshLayout refreshLayout = this.mRefreshLayout;
        if (refreshLayout != null) {
            refreshLayout.autoRefresh();
        }
        LogUtil.e("communityTopicFragment mRefreshLayout.autoRefresh() :" + this.menuId);
    }
}
