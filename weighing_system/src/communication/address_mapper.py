"""
地址映射器
管理参数名称与PLC地址的映射关系
"""

import json
import logging
import os


class AddressMapper:
    """
    地址映射器
    管理参数名称与PLC地址的映射关系
    """
    
    def __init__(self, mapping_file=None):
        """初始化地址映射器
        
        Args:
            mapping_file (str, optional): 地址映射文件路径
        """
        self.logger = logging.getLogger('address_mapper')
        self.mappings = {
            'registers': {},
            'coils': {}
        }
        
        # 基于plc地址.md加载默认映射
        self._load_default_mappings()
        
        # 如果提供了映射文件，则加载它
        if mapping_file and os.path.exists(mapping_file):
            self.load_mapping(mapping_file)
    
    def _load_default_mappings(self):
        """加载默认地址映射"""
        # 数据寄存器 (HD - 32位)
        self.mappings['registers'] = {
            # 称重数据 (读取)
            'weight_data': {
                'addresses': [700, 702, 704, 706, 708, 710],
                'type': 'float32',
                'access': 'read'
            },
            # 粗加料速度
            'coarse_speed': {
                'addresses': [300, 320, 340, 360, 380, 400],
                'type': 'float32',
                'access': 'read_write'
            },
            # 精加料速度
            'fine_speed': {
                'addresses': [302, 322, 342, 362, 382, 402],
                'type': 'float32',
                'access': 'read_write'
            },
            # 粗加提前量
            'coarse_advance': {
                'addresses': [500, 504, 508, 512, 516, 520],
                'type': 'float32',
                'access': 'read_write'
            },
            # 精加提前量
            'fine_advance': {
                'addresses': [502, 506, 510, 514, 518, 522],
                'type': 'float32',
                'access': 'read_write'
            },
            # 目标重量
            'target_weight': {
                'addresses': [141, 142, 143, 144, 145, 146],
                'type': 'float32',
                'access': 'read_write'
            },
            # 点动时间
            'jog_time': {
                'address': 70,
                'type': 'float32',
                'access': 'read_write'
            },
            # 点动间隔时间
            'jog_interval': {
                'address': 72,
                'type': 'float32',
                'access': 'read_write'
            },
            # 清料速度
            'discharge_speed': {
                'address': 290,
                'type': 'float32',
                'access': 'read_write'
            },
            # 清料时间
            'discharge_time': {
                'address': 80,
                'type': 'float32',
                'access': 'read_write'
            },
            # 统一目标重量
            'unified_target_weight': {
                'address': 4,
                'type': 'float32',
                'access': 'read_write'
            }
        }
        
        # 控制线圈 (M地址 - 开关量)
        self.mappings['coils'] = {
            # 总启动
            'master_start': {
                'address': 300,
                'access': 'write'
            },
            # 总停止
            'master_stop': {
                'address': 301,
                'access': 'write'
            },
            # 总清零
            'master_zero': {
                'address': 6,
                'access': 'write'
            },
            # 总放料
            'master_discharge': {
                'address': 5,
                'access': 'write'
            },
            # 总清料
            'master_clean': {
                'address': 7,
                'access': 'write'
            },
            # 斗启动
            'hopper_start': {
                'addresses': [110, 111, 112, 113, 114, 115],
                'access': 'write'
            },
            # 斗停止
            'hopper_stop': {
                'addresses': [120, 121, 122, 123, 124, 125],
                'access': 'write'
            },
            # 斗清零
            'hopper_zero': {
                'addresses': [181, 182, 183, 184, 185, 186],
                'access': 'write'
            },
            # 斗放料
            'hopper_discharge': {
                'addresses': [51, 52, 53, 54, 55, 56],
                'access': 'write'
            },
            # 斗清料
            'hopper_clean': {
                'addresses': [61, 62, 63, 64, 65, 66],
                'access': 'write'
            },
            # 统一重量模式
            'unified_weight_mode': {
                'address': 0,
                'access': 'write'
            }
        }
        
        self.logger.info("已加载默认地址映射")
    
    def load_mapping(self, mapping_file):
        """从文件加载映射关系
        
        Args:
            mapping_file (str): 映射文件路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
                
                # 更新现有映射
                if 'registers' in mappings:
                    self.mappings['registers'].update(mappings['registers'])
                if 'coils' in mappings:
                    self.mappings['coils'].update(mappings['coils'])
                
                self.logger.info(f"已从文件加载地址映射: {mapping_file}")
                return True
        except Exception as e:
            self.logger.error(f"加载地址映射文件时出错: {str(e)}")
            return False
    
    def save_mapping(self, mapping_file):
        """保存映射关系到文件
        
        Args:
            mapping_file (str): 映射文件路径
            
        Returns:
            bool: 保存是否成功
        """
        try:
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.mappings, f, indent=4, ensure_ascii=False)
                self.logger.info(f"已保存地址映射到文件: {mapping_file}")
                return True
        except Exception as e:
            self.logger.error(f"保存地址映射文件时出错: {str(e)}")
            return False
    
    def get_register_address(self, param_name, hopper_id=None):
        """获取寄存器参数对应的PLC地址
        
        Args:
            param_name (str): 参数名称
            hopper_id (int, optional): 料斗ID (0-5)
            
        Returns:
            int: 参数对应的PLC地址，如果未找到返回None
        """
        if param_name not in self.mappings['registers']:
            self.logger.warning(f"未找到寄存器参数: {param_name}")
            return None
            
        param_info = self.mappings['registers'][param_name]
        
        # 如果有多个地址（针对不同料斗）
        if 'addresses' in param_info and hopper_id is not None:
            if 0 <= hopper_id < len(param_info['addresses']):
                return param_info['addresses'][hopper_id]
            else:
                self.logger.warning(f"料斗ID超出范围: {hopper_id}")
                return None
        # 如果是单个地址
        elif 'address' in param_info:
            return param_info['address']
        else:
            self.logger.warning(f"参数{param_name}的地址格式错误")
            return None
    
    def get_coil_address(self, param_name, hopper_id=None):
        """获取线圈参数对应的PLC地址
        
        Args:
            param_name (str): 参数名称
            hopper_id (int, optional): 料斗ID (0-5)
            
        Returns:
            int: 参数对应的PLC地址，如果未找到返回None
        """
        if param_name not in self.mappings['coils']:
            self.logger.warning(f"未找到线圈参数: {param_name}")
            return None
            
        param_info = self.mappings['coils'][param_name]
        
        # 如果有多个地址（针对不同料斗）
        if 'addresses' in param_info and hopper_id is not None:
            if 0 <= hopper_id < len(param_info['addresses']):
                return param_info['addresses'][hopper_id]
            else:
                self.logger.warning(f"料斗ID超出范围: {hopper_id}")
                return None
        # 如果是单个地址
        elif 'address' in param_info:
            return param_info['address']
        else:
            self.logger.warning(f"参数{param_name}的地址格式错误")
            return None
    
    def get_data_type(self, param_name):
        """获取参数的数据类型
        
        Args:
            param_name (str): 参数名称
            
        Returns:
            str: 参数数据类型，如果未找到返回None
        """
        if param_name in self.mappings['registers']:
            return self.mappings['registers'][param_name].get('type')
        elif param_name in self.mappings['coils']:
            return 'bool'
        else:
            self.logger.warning(f"未找到参数: {param_name}")
            return None
    
    def get_access_type(self, param_name):
        """获取参数的访问类型
        
        Args:
            param_name (str): 参数名称
            
        Returns:
            str: 参数访问类型，如果未找到返回None
        """
        if param_name in self.mappings['registers']:
            return self.mappings['registers'][param_name].get('access')
        elif param_name in self.mappings['coils']:
            return self.mappings['coils'][param_name].get('access')
        else:
            self.logger.warning(f"未找到参数: {param_name}")
            return None
            
    def is_register(self, param_name):
        """检查参数是否为寄存器参数
        
        Args:
            param_name (str): 参数名称
            
        Returns:
            bool: 是否为寄存器参数
        """
        return param_name in self.mappings['registers']
        
    def is_coil(self, param_name):
        """检查参数是否为线圈参数
        
        Args:
            param_name (str): 参数名称
            
        Returns:
            bool: 是否为线圈参数
        """
        return param_name in self.mappings['coils']
        
    def get_all_params(self):
        """获取所有参数名称
        
        Returns:
            list: 参数名称列表
        """
        return list(self.mappings['registers'].keys()) + list(self.mappings['coils'].keys()) 