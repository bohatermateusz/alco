package com.hismith.hismithapp.adapter;

import android.bluetooth.BluetoothDevice;
import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.model.BleDeviceModel;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import java.util.List;

/* loaded from: classes3.dex */
public class BluetoothListAdapter extends RecyclerView.Adapter<ViewHolder> {
    private boolean isClicked = false;
    private List<BleDeviceModel> mBluetoothList;
    private Context mContext;
    private OnEditNameListener onEditNameListener;
    private OnItemClickListener onItemClickListener;

    public interface OnEditNameListener {
        void onEditName(BluetoothDevice bluetoothDevice, String str, int i);
    }

    public interface OnItemClickListener {
        void onItemClick(int i);
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private ImageView mBgBattery;
        private ImageView mImgBattery;
        public Button mItemConn;
        public TextView mItemName;
        public TextView mItemTxtRename;
        private ConstraintLayout mLayoutBattery;
        private View mViewBattery;

        public ViewHolder(View view) {
            super(view);
            this.mItemName = (TextView) view.findViewById(R.id.bluetooth_item_txt_name);
            this.mItemConn = (Button) view.findViewById(R.id.bluetooth_item_btn_conn);
            this.mItemTxtRename = (TextView) view.findViewById(R.id.bluetooth_item_txt_rename);
            this.mLayoutBattery = (ConstraintLayout) view.findViewById(R.id.layout_battery);
            this.mViewBattery = view.findViewById(R.id.view_battery);
            this.mImgBattery = (ImageView) view.findViewById(R.id.img_battery);
            this.mBgBattery = (ImageView) view.findViewById(R.id.bg_battery);
            LanguageUtil.setText(BluetoothListAdapter.this.mContext, this.mItemTxtRename, "edit");
        }
    }

    public BluetoothListAdapter(Context context, List<BleDeviceModel> list) {
        this.mContext = context;
        this.mBluetoothList = list;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mBluetoothList.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(final ViewHolder viewHolder, final int i) {
        viewHolder.mItemName.setText(this.mBluetoothList.get(i).getDeviceName());
        viewHolder.mItemName.requestLayout();
        final String deviceAddress = this.mBluetoothList.get(i).getDeviceAddress();
        viewHolder.mItemConn.setAlpha(1.0f);
        viewHolder.mItemConn.setEnabled(true);
        int isConnectedBleProduct = BleManager.getInstance(this.mContext).isConnectedBleProduct(deviceAddress);
        Context context = this.mContext;
        if (isConnectedBleProduct == 2) {
            LanguageUtil.setText(context, viewHolder.mItemConn, "connected");
            viewHolder.mItemConn.setTextColor(this.mContext.getResources().getColor(R.color.app_style_color));
            viewHolder.mItemConn.setBackgroundResource(R.drawable.bg_corners_border_style_15);
            viewHolder.mItemConn.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.BluetoothListAdapter.1
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    BleManager.getInstance(BluetoothListAdapter.this.mContext).disconnectGatt(((BleDeviceModel) BluetoothListAdapter.this.mBluetoothList.get(i)).getBluetoothDevice(), ((BleDeviceModel) BluetoothListAdapter.this.mBluetoothList.get(i)).getDeviceName());
                }
            });
            viewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.BluetoothListAdapter.2
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    if (BluetoothListAdapter.this.onItemClickListener != null) {
                        BluetoothListAdapter.this.onItemClickListener.onItemClick(i);
                    }
                }
            });
            if (BleManager.getInstance(this.mContext).getBleProducts().get(deviceAddress) != null && BleManager.getInstance(this.mContext).getBleProducts().get(deviceAddress).getConnBleDevice() != null && BleManager.getInstance(this.mContext).getBleProducts().get(deviceAddress).getConnBleDevice().getBattery() > -1) {
                if (viewHolder.mLayoutBattery.getVisibility() != 0) {
                    viewHolder.mLayoutBattery.setVisibility(0);
                }
                viewHolder.mViewBattery.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.adapter.BluetoothListAdapter.3
                    @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
                    public void onGlobalLayout() {
                        viewHolder.mViewBattery.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                        ViewGroup.LayoutParams layoutParams = viewHolder.mImgBattery.getLayoutParams();
                        int width = (int) (viewHolder.mViewBattery.getWidth() * (BleManager.getInstance(BluetoothListAdapter.this.mContext).getBleProducts().get(deviceAddress).getConnBleDevice().getBattery() / 10.0d));
                        ViewHolder viewHolder2 = viewHolder;
                        if (width >= 0) {
                            viewHolder2.mImgBattery.setVisibility(0);
                        } else {
                            viewHolder2.mImgBattery.setVisibility(8);
                        }
                        layoutParams.width = width;
                        viewHolder.mImgBattery.setLayoutParams(layoutParams);
                    }
                });
            }
        } else {
            int isConnectedBleProduct2 = BleManager.getInstance(context).isConnectedBleProduct(deviceAddress);
            Context context2 = this.mContext;
            if (isConnectedBleProduct2 == 0) {
                LanguageUtil.setText(context2, viewHolder.mItemConn, "connection");
                viewHolder.mItemConn.setTextColor(this.mContext.getResources().getColor(R.color.white));
                viewHolder.mItemConn.setBackgroundResource(R.drawable.bg_edit_style_15);
                viewHolder.mItemConn.setAlpha(0.2f);
                viewHolder.mItemConn.setEnabled(false);
                if (MyApplication.getApplication().getmProductModel() != null) {
                    viewHolder.mItemConn.setAlpha(1.0f);
                    viewHolder.mItemConn.setEnabled(true);
                    viewHolder.mItemConn.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.BluetoothListAdapter.4
                        @Override // android.view.View.OnClickListener
                        public void onClick(View view) {
                            boolean isExceedMaxConnection = BleManager.getInstance(BluetoothListAdapter.this.mContext).isExceedMaxConnection();
                            BluetoothListAdapter bluetoothListAdapter = BluetoothListAdapter.this;
                            if (isExceedMaxConnection) {
                                ToastUtil.showShortToast(BluetoothListAdapter.this.mContext, LanguageUtil.getLanguageByKey(bluetoothListAdapter.mContext, "maxConnected"));
                            } else {
                                if (bluetoothListAdapter.isClicked) {
                                    return;
                                }
                                BluetoothListAdapter.this.isClicked = true;
                                BluetoothListAdapter.this.notifyItemChanged(i);
                                BleManager.getInstance(BluetoothListAdapter.this.mContext).connectGatt(((BleDeviceModel) BluetoothListAdapter.this.mBluetoothList.get(i)).getBluetoothDevice(), ((BleDeviceModel) BluetoothListAdapter.this.mBluetoothList.get(i)).getDeviceName(), ((BleDeviceModel) BluetoothListAdapter.this.mBluetoothList.get(i)).getDeviceCode());
                            }
                        }
                    });
                } else if (MyApplication.getApplication().getmProductModel() == null) {
                    viewHolder.mItemConn.setAlpha(1.0f);
                    viewHolder.mItemConn.setEnabled(true);
                    viewHolder.mItemConn.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.BluetoothListAdapter.5
                        @Override // android.view.View.OnClickListener
                        public void onClick(View view) {
                            boolean isExceedMaxConnection = BleManager.getInstance(BluetoothListAdapter.this.mContext).isExceedMaxConnection();
                            BluetoothListAdapter bluetoothListAdapter = BluetoothListAdapter.this;
                            if (isExceedMaxConnection) {
                                ToastUtil.showShortToast(BluetoothListAdapter.this.mContext, LanguageUtil.getLanguageByKey(bluetoothListAdapter.mContext, "maxConnected"));
                            } else {
                                if (bluetoothListAdapter.isClicked) {
                                    return;
                                }
                                BluetoothListAdapter.this.isClicked = true;
                                BluetoothListAdapter.this.notifyItemChanged(i);
                                BleManager.getInstance(BluetoothListAdapter.this.mContext).connectGatt(((BleDeviceModel) BluetoothListAdapter.this.mBluetoothList.get(i)).getBluetoothDevice(), ((BleDeviceModel) BluetoothListAdapter.this.mBluetoothList.get(i)).getDeviceName(), ((BleDeviceModel) BluetoothListAdapter.this.mBluetoothList.get(i)).getDeviceCode());
                            }
                        }
                    });
                }
                viewHolder.itemView.setOnClickListener(null);
                viewHolder.mLayoutBattery.setVisibility(8);
            } else if (BleManager.getInstance(context2).isConnectedBleProduct(deviceAddress) == 1) {
                viewHolder.mItemConn.setTextColor(this.mContext.getResources().getColor(R.color.app_style_color));
                viewHolder.mItemConn.setBackgroundResource(R.drawable.bg_corners_border_style_15);
                LanguageUtil.setText(this.mContext, viewHolder.mItemConn, "connecting");
            }
        }
        viewHolder.mItemTxtRename.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.BluetoothListAdapter.6
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (BluetoothListAdapter.this.onEditNameListener != null) {
                    BluetoothListAdapter.this.onEditNameListener.onEditName(((BleDeviceModel) BluetoothListAdapter.this.mBluetoothList.get(i)).getBluetoothDevice(), ((BleDeviceModel) BluetoothListAdapter.this.mBluetoothList.get(i)).getDeviceName(), i);
                }
            }
        });
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new ViewHolder(LayoutInflater.from(viewGroup.getContext()).inflate(R.layout.item_bluetooth, viewGroup, false));
    }

    public void setClicked(boolean z) {
        this.isClicked = z;
    }

    public void setOnEditNameListener(OnEditNameListener onEditNameListener) {
        this.onEditNameListener = onEditNameListener;
    }

    public void setOnItemClickListener(OnItemClickListener onItemClickListener) {
        this.onItemClickListener = onItemClickListener;
    }
}
