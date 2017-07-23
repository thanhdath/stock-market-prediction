# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/framework/step_stats.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from tensorflow.core.framework import allocation_description_pb2 as tensorflow_dot_core_dot_framework_dot_allocation__description__pb2
from tensorflow.core.framework import tensor_description_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__description__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow/core/framework/step_stats.proto',
  package='tensorflow',
  syntax='proto3',
  serialized_pb=_b('\n*tensorflow/core/framework/step_stats.proto\x12\ntensorflow\x1a\x36tensorflow/core/framework/allocation_description.proto\x1a\x32tensorflow/core/framework/tensor_description.proto\"j\n\x13\x41llocatorMemoryUsed\x12\x16\n\x0e\x61llocator_name\x18\x01 \x01(\t\x12\x13\n\x0btotal_bytes\x18\x02 \x01(\x03\x12\x12\n\npeak_bytes\x18\x03 \x01(\x03\x12\x12\n\nlive_bytes\x18\x04 \x01(\x03\"U\n\nNodeOutput\x12\x0c\n\x04slot\x18\x01 \x01(\x05\x12\x39\n\x12tensor_description\x18\x03 \x01(\x0b\x32\x1d.tensorflow.TensorDescription\"\xef\x01\n\x0bMemoryStats\x12\x1d\n\x15host_temp_memory_size\x18\x01 \x01(\x03\x12\x1f\n\x17\x64\x65vice_temp_memory_size\x18\x02 \x01(\x03\x12#\n\x1bhost_persistent_memory_size\x18\x03 \x01(\x03\x12%\n\x1d\x64\x65vice_persistent_memory_size\x18\x04 \x01(\x03\x12(\n host_persistent_tensor_alloc_ids\x18\x05 \x03(\x03\x12*\n\"device_persistent_tensor_alloc_ids\x18\x06 \x03(\x03\"\x9b\x03\n\rNodeExecStats\x12\x11\n\tnode_name\x18\x01 \x01(\t\x12\x18\n\x10\x61ll_start_micros\x18\x02 \x01(\x03\x12\x1b\n\x13op_start_rel_micros\x18\x03 \x01(\x03\x12\x19\n\x11op_end_rel_micros\x18\x04 \x01(\x03\x12\x1a\n\x12\x61ll_end_rel_micros\x18\x05 \x01(\x03\x12/\n\x06memory\x18\x06 \x03(\x0b\x32\x1f.tensorflow.AllocatorMemoryUsed\x12&\n\x06output\x18\x07 \x03(\x0b\x32\x16.tensorflow.NodeOutput\x12\x16\n\x0etimeline_label\x18\x08 \x01(\t\x12\x18\n\x10scheduled_micros\x18\t \x01(\x03\x12\x11\n\tthread_id\x18\n \x01(\r\x12<\n\x11referenced_tensor\x18\x0b \x03(\x0b\x32!.tensorflow.AllocationDescription\x12-\n\x0cmemory_stats\x18\x0c \x01(\x0b\x32\x17.tensorflow.MemoryStats\"P\n\x0f\x44\x65viceStepStats\x12\x0e\n\x06\x64\x65vice\x18\x01 \x01(\t\x12-\n\nnode_stats\x18\x02 \x03(\x0b\x32\x19.tensorflow.NodeExecStats\";\n\tStepStats\x12.\n\tdev_stats\x18\x01 \x03(\x0b\x32\x1b.tensorflow.DeviceStepStatsB0\n\x18org.tensorflow.frameworkB\x0fStepStatsProtosP\x01\xf8\x01\x01\x62\x06proto3')
  ,
  dependencies=[tensorflow_dot_core_dot_framework_dot_allocation__description__pb2.DESCRIPTOR,tensorflow_dot_core_dot_framework_dot_tensor__description__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_ALLOCATORMEMORYUSED = _descriptor.Descriptor(
  name='AllocatorMemoryUsed',
  full_name='tensorflow.AllocatorMemoryUsed',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='allocator_name', full_name='tensorflow.AllocatorMemoryUsed.allocator_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='total_bytes', full_name='tensorflow.AllocatorMemoryUsed.total_bytes', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='peak_bytes', full_name='tensorflow.AllocatorMemoryUsed.peak_bytes', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='live_bytes', full_name='tensorflow.AllocatorMemoryUsed.live_bytes', index=3,
      number=4, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=166,
  serialized_end=272,
)


_NODEOUTPUT = _descriptor.Descriptor(
  name='NodeOutput',
  full_name='tensorflow.NodeOutput',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='slot', full_name='tensorflow.NodeOutput.slot', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='tensor_description', full_name='tensorflow.NodeOutput.tensor_description', index=1,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=274,
  serialized_end=359,
)


_MEMORYSTATS = _descriptor.Descriptor(
  name='MemoryStats',
  full_name='tensorflow.MemoryStats',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='host_temp_memory_size', full_name='tensorflow.MemoryStats.host_temp_memory_size', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='device_temp_memory_size', full_name='tensorflow.MemoryStats.device_temp_memory_size', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='host_persistent_memory_size', full_name='tensorflow.MemoryStats.host_persistent_memory_size', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='device_persistent_memory_size', full_name='tensorflow.MemoryStats.device_persistent_memory_size', index=3,
      number=4, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='host_persistent_tensor_alloc_ids', full_name='tensorflow.MemoryStats.host_persistent_tensor_alloc_ids', index=4,
      number=5, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='device_persistent_tensor_alloc_ids', full_name='tensorflow.MemoryStats.device_persistent_tensor_alloc_ids', index=5,
      number=6, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=362,
  serialized_end=601,
)


_NODEEXECSTATS = _descriptor.Descriptor(
  name='NodeExecStats',
  full_name='tensorflow.NodeExecStats',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='node_name', full_name='tensorflow.NodeExecStats.node_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='all_start_micros', full_name='tensorflow.NodeExecStats.all_start_micros', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='op_start_rel_micros', full_name='tensorflow.NodeExecStats.op_start_rel_micros', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='op_end_rel_micros', full_name='tensorflow.NodeExecStats.op_end_rel_micros', index=3,
      number=4, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='all_end_rel_micros', full_name='tensorflow.NodeExecStats.all_end_rel_micros', index=4,
      number=5, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='memory', full_name='tensorflow.NodeExecStats.memory', index=5,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='output', full_name='tensorflow.NodeExecStats.output', index=6,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='timeline_label', full_name='tensorflow.NodeExecStats.timeline_label', index=7,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='scheduled_micros', full_name='tensorflow.NodeExecStats.scheduled_micros', index=8,
      number=9, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='thread_id', full_name='tensorflow.NodeExecStats.thread_id', index=9,
      number=10, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='referenced_tensor', full_name='tensorflow.NodeExecStats.referenced_tensor', index=10,
      number=11, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='memory_stats', full_name='tensorflow.NodeExecStats.memory_stats', index=11,
      number=12, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=604,
  serialized_end=1015,
)


_DEVICESTEPSTATS = _descriptor.Descriptor(
  name='DeviceStepStats',
  full_name='tensorflow.DeviceStepStats',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='device', full_name='tensorflow.DeviceStepStats.device', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='node_stats', full_name='tensorflow.DeviceStepStats.node_stats', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1017,
  serialized_end=1097,
)


_STEPSTATS = _descriptor.Descriptor(
  name='StepStats',
  full_name='tensorflow.StepStats',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dev_stats', full_name='tensorflow.StepStats.dev_stats', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1099,
  serialized_end=1158,
)

_NODEOUTPUT.fields_by_name['tensor_description'].message_type = tensorflow_dot_core_dot_framework_dot_tensor__description__pb2._TENSORDESCRIPTION
_NODEEXECSTATS.fields_by_name['memory'].message_type = _ALLOCATORMEMORYUSED
_NODEEXECSTATS.fields_by_name['output'].message_type = _NODEOUTPUT
_NODEEXECSTATS.fields_by_name['referenced_tensor'].message_type = tensorflow_dot_core_dot_framework_dot_allocation__description__pb2._ALLOCATIONDESCRIPTION
_NODEEXECSTATS.fields_by_name['memory_stats'].message_type = _MEMORYSTATS
_DEVICESTEPSTATS.fields_by_name['node_stats'].message_type = _NODEEXECSTATS
_STEPSTATS.fields_by_name['dev_stats'].message_type = _DEVICESTEPSTATS
DESCRIPTOR.message_types_by_name['AllocatorMemoryUsed'] = _ALLOCATORMEMORYUSED
DESCRIPTOR.message_types_by_name['NodeOutput'] = _NODEOUTPUT
DESCRIPTOR.message_types_by_name['MemoryStats'] = _MEMORYSTATS
DESCRIPTOR.message_types_by_name['NodeExecStats'] = _NODEEXECSTATS
DESCRIPTOR.message_types_by_name['DeviceStepStats'] = _DEVICESTEPSTATS
DESCRIPTOR.message_types_by_name['StepStats'] = _STEPSTATS

AllocatorMemoryUsed = _reflection.GeneratedProtocolMessageType('AllocatorMemoryUsed', (_message.Message,), dict(
  DESCRIPTOR = _ALLOCATORMEMORYUSED,
  __module__ = 'tensorflow.core.framework.step_stats_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.AllocatorMemoryUsed)
  ))
_sym_db.RegisterMessage(AllocatorMemoryUsed)

NodeOutput = _reflection.GeneratedProtocolMessageType('NodeOutput', (_message.Message,), dict(
  DESCRIPTOR = _NODEOUTPUT,
  __module__ = 'tensorflow.core.framework.step_stats_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.NodeOutput)
  ))
_sym_db.RegisterMessage(NodeOutput)

MemoryStats = _reflection.GeneratedProtocolMessageType('MemoryStats', (_message.Message,), dict(
  DESCRIPTOR = _MEMORYSTATS,
  __module__ = 'tensorflow.core.framework.step_stats_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.MemoryStats)
  ))
_sym_db.RegisterMessage(MemoryStats)

NodeExecStats = _reflection.GeneratedProtocolMessageType('NodeExecStats', (_message.Message,), dict(
  DESCRIPTOR = _NODEEXECSTATS,
  __module__ = 'tensorflow.core.framework.step_stats_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.NodeExecStats)
  ))
_sym_db.RegisterMessage(NodeExecStats)

DeviceStepStats = _reflection.GeneratedProtocolMessageType('DeviceStepStats', (_message.Message,), dict(
  DESCRIPTOR = _DEVICESTEPSTATS,
  __module__ = 'tensorflow.core.framework.step_stats_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.DeviceStepStats)
  ))
_sym_db.RegisterMessage(DeviceStepStats)

StepStats = _reflection.GeneratedProtocolMessageType('StepStats', (_message.Message,), dict(
  DESCRIPTOR = _STEPSTATS,
  __module__ = 'tensorflow.core.framework.step_stats_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.StepStats)
  ))
_sym_db.RegisterMessage(StepStats)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\n\030org.tensorflow.frameworkB\017StepStatsProtosP\001\370\001\001'))
# @@protoc_insertion_point(module_scope)
