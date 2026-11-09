"""Static task semantics. Parse source AST; never execute downloaded task code."""
import ast,re,hashlib
from pathlib import Path

def infer_schema(source):
 text=Path(source).read_text();tree=ast.parse(text)
 play=next((n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='play_once'),None)
 if play is None:return {'supported':False,'reason':'missing_play_once'}
 calls=sorted([n for n in ast.walk(play) if isinstance(n,ast.Call)],key=lambda n:(n.lineno,n.col_offset))
 grasp=[];place=[]
 for n in calls:
  if isinstance(n.func,ast.Attribute) and n.func.attr in ('grasp_actor','place_actor'):
   value=n.args[0] if n.args else next((k.value for k in n.keywords if k.arg=='actor'),None)
   actor=ast.unparse(value) if value is not None else 'unknown'
   (grasp if n.func.attr=='grasp_actor' else place).append(actor)
 # Bind a task variable to the scene_info placeholder only for an exact model-name
 # expression match; otherwise retain an unresolved binding, never invent one.
 model_expr={};models={};placeholders={}
 for n in ast.walk(tree):
  if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id in ('create_actor','create_box','create_cylinder'):
   kwargs={k.arg:k.value for k in n.value.keywords}
   for target in n.targets:
    var=ast.unparse(target)
    if 'modelname' in kwargs:
     expr=kwargs['modelname'];model_expr[var]=ast.unparse(expr)
     if isinstance(expr,ast.Constant) and isinstance(expr.value,str):models[var]=expr.value
  if isinstance(n,ast.Dict):
   for k,v in zip(n.keys,n.values):
    if isinstance(k,ast.Constant) and isinstance(k.value,str) and re.fullmatch(r'\{[A-Z]\}',k.value):
     placeholders[k.value]=ast.unparse(v)
 bindings={}
 for actor,expr in model_expr.items():
  matches=[k for k,v in placeholders.items() if (actor in models and models[actor] in v) or (expr.startswith('self.') and '{'+expr+'}' in v)]
  if len(matches)==1:bindings[actor]=matches[0]
 return {'source_sha256':hashlib.sha256(text.encode()).hexdigest(),'grasp_actors':list(dict.fromkeys(grasp)),
         'place_actors':list(dict.fromkeys(place)),'bindings':bindings,'models':models,
         'has_branches':any(isinstance(n,(ast.If,ast.For,ast.While)) for n in ast.walk(play)),
         'source_file':str(source),'target_symbol':next((ast.unparse(k.value).replace('self.','').replace('.','_') for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='place_actor' for k in n.keywords if k.arg=='target_pose' and isinstance(k.value,ast.Attribute)),None)}

def object_for(schema,scene):
 actors=schema.get('grasp_actors',[])
 # Only unique-object tasks have automatic object bindings. Multiple-object and
 # repeated grasp programs require dynamic traces or reviewed sidecars.
 if len(actors)!=1:return None,'multiple_or_missing_grasp_objects'
 actor=actors[0];key=schema.get('bindings',{}).get(actor)
 if key and isinstance(scene.get('info',{}).get(key),str):return scene['info'][key],None
 return None,'unresolved_scene_object_binding'
