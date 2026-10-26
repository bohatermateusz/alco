package com.hismith.hismithapp.popup;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.popup.CommonPopupWindow;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.hismith.hismithapp.view.decoration.DividerItemDecoration;
import java.util.List;

/* loaded from: classes3.dex */
public class ActionAttachPopup2 {
    private ListAdapter mAdapter;
    private Context mContext;
    private FrameLayout mLayout;
    private LinearLayoutManager mLayoutManager;
    private List<ActionItem2> mList;
    private View.OnClickListener mOnCollectClickListener;
    private View.OnClickListener mOnCopyClickListener;
    private View.OnClickListener mOnCycleClickListener;
    private View.OnClickListener mOnDelClickListener;
    private View.OnClickListener mOnEditClickListener;
    private View.OnClickListener mOnRenameClickListener;
    private View.OnClickListener mOnReportClickListener;
    private View.OnClickListener mOnShareClickListener;
    private View.OnClickListener mOnSingleClickListener;
    private RecyclerView mRecyclerView;
    private CommonPopupWindow popupWindow;
    private View viewGroup;
    private int windowWidth;

    public static class ActionItem2 {
        int icon;
        String text;

        public ActionItem2(int i, String str) {
            this.icon = i;
            this.text = str;
        }
    }

    class ListAdapter extends RecyclerView.Adapter<ViewHolder> {

        class ViewHolder extends RecyclerView.ViewHolder {
            private ImageView mIcon;
            private View mLine;
            private TextView mName;

            public ViewHolder(View view) {
                super(view);
                this.mIcon = (ImageView) view.findViewById(R.id.popup_img);
                this.mName = (TextView) view.findViewById(R.id.popup_txt);
                this.mLine = view.findViewById(R.id.popup_line);
            }
        }

        ListAdapter() {
        }

        @Override // androidx.recyclerview.widget.RecyclerView.Adapter
        public int getItemCount() {
            return ActionAttachPopup2.this.mList.size();
        }

        @Override // androidx.recyclerview.widget.RecyclerView.Adapter
        public /* bridge */ /* synthetic */ void onBindViewHolder(ViewHolder viewHolder, int i, List list) {
            onBindViewHolder2(viewHolder, i, (List<Object>) list);
        }

        @Override // androidx.recyclerview.widget.RecyclerView.Adapter
        public void onBindViewHolder(ViewHolder viewHolder, final int i) {
            viewHolder.mIcon.setImageResource(((ActionItem2) ActionAttachPopup2.this.mList.get(i)).icon);
            LanguageUtil.setText(ActionAttachPopup2.this.mContext, viewHolder.mName, ((ActionItem2) ActionAttachPopup2.this.mList.get(i)).text);
            viewHolder.mLine.setVisibility(8);
            viewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.popup.ActionAttachPopup2.ListAdapter.1
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    if (ActionAttachPopup2.this.popupWindow != null && ActionAttachPopup2.this.popupWindow.isShowing()) {
                        ActionAttachPopup2.this.popupWindow.dismiss();
                    }
                    if ("addCollect".equals(((ActionItem2) ActionAttachPopup2.this.mList.get(i)).text) || "uncollect".equals(((ActionItem2) ActionAttachPopup2.this.mList.get(i)).text)) {
                        if (ActionAttachPopup2.this.mOnCollectClickListener != null) {
                            ActionAttachPopup2.this.mOnCollectClickListener.onClick(view);
                            return;
                        }
                        return;
                    }
                    boolean equals = "report".equals(((ActionItem2) ActionAttachPopup2.this.mList.get(i)).text);
                    ListAdapter listAdapter = ListAdapter.this;
                    if (equals) {
                        if (ActionAttachPopup2.this.mOnReportClickListener != null) {
                            ActionAttachPopup2.this.mOnReportClickListener.onClick(view);
                            return;
                        }
                        return;
                    }
                    boolean equals2 = "modeShare".equals(((ActionItem2) ActionAttachPopup2.this.mList.get(i)).text);
                    ListAdapter listAdapter2 = ListAdapter.this;
                    if (equals2) {
                        if (ActionAttachPopup2.this.mOnShareClickListener != null) {
                            ActionAttachPopup2.this.mOnShareClickListener.onClick(view);
                            return;
                        }
                        return;
                    }
                    boolean equals3 = "rename".equals(((ActionItem2) ActionAttachPopup2.this.mList.get(i)).text);
                    ListAdapter listAdapter3 = ListAdapter.this;
                    if (equals3) {
                        if (ActionAttachPopup2.this.mOnRenameClickListener != null) {
                            ActionAttachPopup2.this.mOnRenameClickListener.onClick(view);
                            return;
                        }
                        return;
                    }
                    boolean equals4 = "delete".equals(((ActionItem2) ActionAttachPopup2.this.mList.get(i)).text);
                    ListAdapter listAdapter4 = ListAdapter.this;
                    if (equals4) {
                        if (ActionAttachPopup2.this.mOnDelClickListener != null) {
                            ActionAttachPopup2.this.mOnDelClickListener.onClick(view);
                            return;
                        }
                        return;
                    }
                    boolean equals5 = "edit".equals(((ActionItem2) ActionAttachPopup2.this.mList.get(i)).text);
                    ListAdapter listAdapter5 = ListAdapter.this;
                    if (equals5) {
                        if (ActionAttachPopup2.this.mOnEditClickListener != null) {
                            ActionAttachPopup2.this.mOnEditClickListener.onClick(view);
                        }
                    } else {
                        if (!"copy".equals(((ActionItem2) ActionAttachPopup2.this.mList.get(i)).text) || ActionAttachPopup2.this.mOnCopyClickListener == null) {
                            return;
                        }
                        ActionAttachPopup2.this.mOnCopyClickListener.onClick(view);
                    }
                }
            });
        }

        /* renamed from: onBindViewHolder, reason: avoid collision after fix types in other method */
        public void onBindViewHolder2(ViewHolder viewHolder, int i, List<Object> list) {
            if (list.isEmpty()) {
                onBindViewHolder(viewHolder, i);
            } else {
                viewHolder.mIcon.setImageResource(((ActionItem2) ActionAttachPopup2.this.mList.get(i)).icon);
                LanguageUtil.setText(ActionAttachPopup2.this.mContext, viewHolder.mName, ((ActionItem2) ActionAttachPopup2.this.mList.get(i)).text);
            }
        }

        @Override // androidx.recyclerview.widget.RecyclerView.Adapter
        public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
            return new ViewHolder(LayoutInflater.from(ActionAttachPopup2.this.mContext).inflate(R.layout.item_play_mode_popup, viewGroup, false));
        }
    }

    public ActionAttachPopup2(Context context, List<ActionItem2> list) {
        this.mContext = context;
        this.mList = list;
        initView();
        initData();
        setListener();
    }

    private static int[] calculatePopWindowPos(View view, View view2) {
        int[] iArr = new int[3];
        int[] iArr2 = new int[2];
        view.getLocationOnScreen(iArr2);
        int width = view.getWidth();
        int height = view.getHeight();
        int screenHeight = MeasureUtil.getScreenHeight(view.getContext());
        MeasureUtil.getScreenWidth(view.getContext());
        view2.measure(0, 0);
        int measuredHeight = view2.getMeasuredHeight();
        int measuredWidth = view2.getMeasuredWidth();
        int i = iArr2[1];
        if ((screenHeight - i) - height < measuredHeight) {
            iArr[0] = (iArr2[0] - measuredWidth) + width;
            iArr[1] = i - measuredHeight;
            iArr[2] = 1;
            return iArr;
        }
        iArr[0] = (iArr2[0] - measuredWidth) + width;
        iArr[1] = i + height;
        iArr[2] = 0;
        return iArr;
    }

    private void initData() {
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this.mContext);
        this.mLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mRecyclerView.addItemDecoration(new DividerItemDecoration(this.mContext, 1));
        ListAdapter listAdapter = new ListAdapter();
        this.mAdapter = listAdapter;
        this.mRecyclerView.setAdapter(listAdapter);
    }

    private void initView() {
        View inflate = LayoutInflater.from(this.mContext).inflate(R.layout.dialog_action_attach_list, (ViewGroup) null, false);
        this.viewGroup = inflate;
        this.mLayout = (FrameLayout) inflate.findViewById(R.id.dialog_layout);
        this.mRecyclerView = (RecyclerView) this.viewGroup.findViewById(R.id.dialog_recycler_view);
    }

    private void setListener() {
    }

    public void dissPop() {
        CommonPopupWindow commonPopupWindow = this.popupWindow;
        if (commonPopupWindow != null) {
            commonPopupWindow.dismiss();
        }
    }

    public void notifyDataSetChanged() {
        ListAdapter listAdapter = this.mAdapter;
        if (listAdapter != null) {
            listAdapter.notifyDataSetChanged();
        }
    }

    public void notifyItemChanged(int i) {
        ListAdapter listAdapter = this.mAdapter;
        if (listAdapter != null) {
            listAdapter.notifyItemChanged(i, "");
        }
    }

    public void setCollectClickListener(View.OnClickListener onClickListener) {
        this.mOnCollectClickListener = onClickListener;
    }

    public void setOnCopyClickListener(View.OnClickListener onClickListener) {
        this.mOnCopyClickListener = onClickListener;
    }

    public void setOnCycleClickListener(View.OnClickListener onClickListener) {
        this.mOnCycleClickListener = onClickListener;
    }

    public void setOnDelClickListener(View.OnClickListener onClickListener) {
        this.mOnDelClickListener = onClickListener;
    }

    public void setOnEditClickListener(View.OnClickListener onClickListener) {
        this.mOnEditClickListener = onClickListener;
    }

    public void setOnRenameClickListener(View.OnClickListener onClickListener) {
        this.mOnRenameClickListener = onClickListener;
    }

    public void setOnReportClickListener(View.OnClickListener onClickListener) {
        this.mOnReportClickListener = onClickListener;
    }

    public void setOnShareClickListener(View.OnClickListener onClickListener) {
        this.mOnShareClickListener = onClickListener;
    }

    public void setOnSingleClickListener(View.OnClickListener onClickListener) {
        this.mOnSingleClickListener = onClickListener;
    }

    public void showPop(View view, int i) {
        CommonPopupWindow commonPopupWindow = this.popupWindow;
        if (commonPopupWindow == null || !commonPopupWindow.isShowing()) {
            this.popupWindow = new CommonPopupWindow.Builder(this.mContext).setView(this.viewGroup).setWidthAndHeight(-2, -2).setOutsideTouchable(true).create();
            int[] calculatePopWindowPos = calculatePopWindowPos(view, this.viewGroup);
            int i2 = calculatePopWindowPos[2];
            CommonPopupWindow commonPopupWindow2 = this.popupWindow;
            if (i2 == 1) {
                commonPopupWindow2.setAnimationStyle(R.style.AnimUp);
                this.mLayout.setBackground(this.mContext.getResources().getDrawable(R.drawable.bg_play_mode_action));
                calculatePopWindowPos[1] = calculatePopWindowPos[1] + MeasureUtil.dip2px(this.mContext, 5.0f);
            } else {
                commonPopupWindow2.setAnimationStyle(R.style.popmenu_animation);
                this.mLayout.setBackground(this.mContext.getResources().getDrawable(R.drawable.bg_play_mode_action));
                calculatePopWindowPos[1] = calculatePopWindowPos[1] - MeasureUtil.dip2px(this.mContext, 5.0f);
            }
            this.popupWindow.showAtLocation(view, 0, calculatePopWindowPos[0], calculatePopWindowPos[1]);
        }
    }
}
