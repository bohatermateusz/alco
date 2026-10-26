package com.hismith.hismithapp.popup;

import android.content.Context;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewpager.widget.PagerAdapter;
import androidx.viewpager.widget.ViewPager;
import com.facebook.appevents.internal.ViewHierarchyConstants;
import com.google.android.material.tabs.TabLayout;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.model.PlayModeListModel;
import com.hismith.hismithapp.model.PlayModeModel;
import com.hismith.hismithapp.popup.CommonPopupWindow;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
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
public class PagerDrawerPopup2 {
    private RetrofitRequest baseRequest;
    private Context mContext;
    private ViewPager pager;
    private CommonPopupWindow popupWindow;
    private TabLayout tabLayout;
    private List<String> titles;
    private TextView txtTitle;
    private View viewGroup;
    private int windowWidth;
    private String[] mRequestTypes = {ViewHierarchyConstants.DIMENSION_TOP_KEY, "new", "my", "collect", "history"};
    private String[] mRequestIds = new String[5];
    private List<ListAdapter> mListAdapters = new ArrayList();
    private List<List<PlayModeModel>> mModesList = new ArrayList();

    private class ListAdapter extends RecyclerView.Adapter<ViewHolder> {
        private List<PlayModeModel> mList;

        class ViewHolder extends RecyclerView.ViewHolder {
            private LinearLayout mLayout;
            private View mLineBottom;
            private View mLineTop;
            private TextView mTxtName;
            private TextView mTxtTime;

            public ViewHolder(View view) {
                super(view);
                this.mLayout = (LinearLayout) view.findViewById(R.id.line_mode_layout);
                this.mTxtName = (TextView) view.findViewById(R.id.line_mode_txt_name);
                this.mTxtTime = (TextView) view.findViewById(R.id.line_mode_txt_time);
                this.mLineTop = view.findViewById(R.id.line_mode_line_top);
                this.mLineBottom = view.findViewById(R.id.line_mode_line_bottom);
            }
        }

        public ListAdapter(List<PlayModeModel> list) {
            this.mList = list;
        }

        @Override // androidx.recyclerview.widget.RecyclerView.Adapter
        public int getItemCount() {
            return this.mList.size();
        }

        public List<PlayModeModel> getList() {
            return this.mList;
        }

        @Override // androidx.recyclerview.widget.RecyclerView.Adapter
        public void onBindViewHolder(ViewHolder viewHolder, final int i) {
            if (i == 0) {
                viewHolder.mLineTop.setVisibility(8);
            } else {
                viewHolder.mLineTop.setVisibility(0);
            }
            if (i == this.mList.size() - 1) {
                viewHolder.mLineBottom.setVisibility(8);
            } else {
                viewHolder.mLineBottom.setVisibility(0);
            }
            viewHolder.mTxtName.setText(this.mList.get(i).getModeName());
            viewHolder.mTxtTime.setText(this.mList.get(i).getModeTime());
            viewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.popup.PagerDrawerPopup2.ListAdapter.1
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    SocketManager.getInstance().roomSetMode(((PlayModeModel) ListAdapter.this.mList.get(i)).getModeId(), 0);
                    if (PagerDrawerPopup2.this.popupWindow != null) {
                        PagerDrawerPopup2.this.popupWindow.dismiss();
                    }
                }
            });
        }

        @Override // androidx.recyclerview.widget.RecyclerView.Adapter
        public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
            return new ViewHolder(LayoutInflater.from(PagerDrawerPopup2.this.mContext).inflate(R.layout.item_line_mode, viewGroup, false));
        }
    }

    class PAdapter extends PagerAdapter {
        PAdapter() {
        }

        @Override // androidx.viewpager.widget.PagerAdapter
        public void destroyItem(ViewGroup viewGroup, int i, Object obj) {
            viewGroup.removeView((View) obj);
        }

        @Override // androidx.viewpager.widget.PagerAdapter
        public int getCount() {
            return PagerDrawerPopup2.this.titles.size();
        }

        @Override // androidx.viewpager.widget.PagerAdapter
        public CharSequence getPageTitle(int i) {
            return (CharSequence) PagerDrawerPopup2.this.titles.get(i);
        }

        @Override // androidx.viewpager.widget.PagerAdapter
        public Object instantiateItem(ViewGroup viewGroup, final int i) {
            LogUtil.e("instantiateItem：" + i + ", " + viewGroup);
            View inflate = LayoutInflater.from(viewGroup.getContext()).inflate(R.layout.view_refresh_layout, viewGroup, false);
            SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) inflate.findViewById(R.id.refreshLayout);
            RecyclerView recyclerView = (RecyclerView) inflate.findViewById(R.id.recyclerView);
            recyclerView.setLayoutManager(new LinearLayoutManager(PagerDrawerPopup2.this.mContext));
            ArrayList arrayList = new ArrayList();
            ListAdapter listAdapter = PagerDrawerPopup2.this.new ListAdapter(arrayList);
            recyclerView.setAdapter(listAdapter);
            PagerDrawerPopup2.this.mListAdapters.add(listAdapter);
            PagerDrawerPopup2.this.mModesList.add(arrayList);
            smartRefreshLayout.setOnRefreshLoadMoreListener(new OnRefreshLoadMoreListener() { // from class: com.hismith.hismithapp.popup.PagerDrawerPopup2.PAdapter.1
                @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
                public void onLoadMore(RefreshLayout refreshLayout) {
                    PagerDrawerPopup2 pagerDrawerPopup2 = PagerDrawerPopup2.this;
                    String str = PagerDrawerPopup2.this.mRequestTypes[i];
                    String[] strArr = PagerDrawerPopup2.this.mRequestIds;
                    int i2 = i;
                    pagerDrawerPopup2.playModeList(str, 0, strArr[i2], refreshLayout, i2);
                }

                @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
                public void onRefresh(RefreshLayout refreshLayout) {
                    PagerDrawerPopup2.this.mRequestIds[i] = "";
                    PagerDrawerPopup2 pagerDrawerPopup2 = PagerDrawerPopup2.this;
                    String str = PagerDrawerPopup2.this.mRequestTypes[i];
                    String[] strArr = PagerDrawerPopup2.this.mRequestIds;
                    int i2 = i;
                    pagerDrawerPopup2.playModeList(str, 1, strArr[i2], refreshLayout, i2);
                }
            });
            smartRefreshLayout.autoRefresh();
            viewGroup.addView(inflate);
            return inflate;
        }

        @Override // androidx.viewpager.widget.PagerAdapter
        public boolean isViewFromObject(View view, Object obj) {
            return view == obj;
        }
    }

    public PagerDrawerPopup2(Context context) {
        this.mContext = context;
        initView();
        initData();
        setListener();
    }

    private void initData() {
        this.baseRequest = new RetrofitRequest();
        ArrayList arrayList = new ArrayList();
        this.titles = arrayList;
        arrayList.add(LanguageUtil.getLanguageByKey(this.mContext, "modeMenuTop"));
        this.titles.add(LanguageUtil.getLanguageByKey(this.mContext, "modeMenuNew"));
        this.titles.add(LanguageUtil.getLanguageByKey(this.mContext, "modeMenuMy"));
        this.titles.add(LanguageUtil.getLanguageByKey(this.mContext, "modeMenuHistory"));
        this.titles.add(LanguageUtil.getLanguageByKey(this.mContext, "modeMenuCollect"));
        this.pager.setOffscreenPageLimit(this.titles.size());
        this.pager.setAdapter(new PAdapter());
        this.tabLayout.setupWithViewPager(this.pager);
    }

    private void initView() {
        View inflate = LayoutInflater.from(this.mContext).inflate(R.layout.dialog_pager_drawer, (ViewGroup) null, false);
        this.viewGroup = inflate;
        TextView textView = (TextView) inflate.findViewById(R.id.txtTitle);
        this.txtTitle = textView;
        LanguageUtil.setText(this.mContext, textView, "playMode");
        this.tabLayout = (TabLayout) this.viewGroup.findViewById(R.id.tabLayout);
        this.pager = (ViewPager) this.viewGroup.findViewById(R.id.pager);
        this.viewGroup.measure(0, 0);
        this.windowWidth = this.viewGroup.getMeasuredWidth();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void playModeList(String str, int i, final String str2, final RefreshLayout refreshLayout, final int i2) {
        this.baseRequest.getApi().playModeList((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), str, MyApplication.getApplication().getDeviceCode(), i, str2).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<PlayModeListModel>(this.mContext, null, false) { // from class: com.hismith.hismithapp.popup.PagerDrawerPopup2.1
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                refreshLayout.finishRefresh();
                refreshLayout.finishLoadMore();
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                refreshLayout.finishRefresh();
                refreshLayout.finishLoadMore();
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(PlayModeListModel playModeListModel) {
                if (playModeListModel.getSuccess() == 1) {
                    if (TextUtils.isEmpty(str2)) {
                        ((List) PagerDrawerPopup2.this.mModesList.get(i2)).clear();
                        ((ListAdapter) PagerDrawerPopup2.this.mListAdapters.get(i2)).notifyDataSetChanged();
                    }
                    if (!TextUtils.isEmpty(playModeListModel.getRequestId())) {
                        PagerDrawerPopup2.this.mRequestIds[i2] = playModeListModel.getRequestId();
                    }
                    if (playModeListModel.getPlayModes() == null || playModeListModel.getPlayModes().length <= 0) {
                        return;
                    }
                    ((List) PagerDrawerPopup2.this.mModesList.get(i2)).addAll(Arrays.asList(playModeListModel.getPlayModes()));
                    ((ListAdapter) PagerDrawerPopup2.this.mListAdapters.get(i2)).notifyDataSetChanged();
                }
            }
        });
    }

    private void setListener() {
    }

    public void dissPop() {
        CommonPopupWindow commonPopupWindow = this.popupWindow;
        if (commonPopupWindow != null) {
            commonPopupWindow.dismiss();
        }
    }

    public void showDownPop(View view) {
        CommonPopupWindow commonPopupWindow = this.popupWindow;
        if (commonPopupWindow == null || !commonPopupWindow.isShowing()) {
            this.popupWindow = new CommonPopupWindow.Builder(this.mContext).setView(this.viewGroup).setWidthAndHeight(-1, -1).setAnimationStyle(R.style.popmenu_animation).setOutsideTouchable(true).create();
            int[] iArr = new int[2];
            view.getLocationOnScreen(iArr);
            this.popupWindow.showAtLocation(view, 0, (iArr[0] - this.windowWidth) - MeasureUtil.dip2px(this.mContext, 20.0f), iArr[1] + view.getHeight());
        }
    }

    public void showPopwindow() {
        CommonPopupWindow create = new CommonPopupWindow.Builder(this.mContext).setView(this.viewGroup).setWidthAndHeight(-2, -2).setAnimationStyle(R.style.popmenu_animation).setOutsideTouchable(true).create();
        this.popupWindow = create;
        create.showAtLocation(this.viewGroup, 80, 0, 0);
    }
}
