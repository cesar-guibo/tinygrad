self.opts = Device[Device.DEFAULT].renderer
self.uops:List[UOp] = linearize_uop(full_graph_rewrite(rewrite_shapetracker_with_index(modified_ast, self.opts), self.opts))